// CortexModelServer: from CortexAgent (crates/llama-engine/native/gc_engine.cpp), hosted over
// loopback HTTP the way upstream llama.cpp's tools/server/server.cpp hosts one model.
//
// CortexAgent runs llama.cpp's own server core (server_context and server_routes, from its vendored
// unslothai/llama.cpp 468ad58 (b11876) with the four patch sites in vendor/llama.cpp/PATCHES.md)
// in-process, with no HTTP server.
// Its vendored tree does not contain upstream's server.cpp, so this file is the executable's entry:
//
// - arguments are parsed by llama.cpp's own parser as LLAMA_EXAMPLE_SERVER (as gc_engine and
//   llama-server both do), so `--list-devices` and every flag behave as in llama-server;
// - the model is loaded and the task loop runs exactly as in gc_engine / server.cpp's instance
//   mode (load_model, update_meta, start_loop);
// - the routes are the ones gc_engine exposes (health, slots, slot save/restore/erase, chat
//   completions and its input-token count, completion, apply-template, tokenize, detokenize, and
//   gc_engine's own /tool-call-markers), plus the read-only ones the plugin and the frame read
//   (/v1/models, /props, /metrics), each wrapped with upstream's exception-to-response mapping.
//
// Not carried over from upstream's server.cpp (468ad58): router mode, the download worker, -hf /
// docker model resolution, llama-connect, the MCP proxy, server tools and the resumable-stream
// routes (CortexAgent uses none of them; they stay off without their flags). Its
// server_prepare_params is reduced to the model alias: the embeddings batch fix, the auto
// --parallel and --kv-unified-per-slot sizing do not apply to CortexAgent's explicit arguments
// (--parallel N --ctx-size C), and gc_engine runs none of it.

#include "server-context.h"
#include "server-common.h"
#include "server-http.h"
#include "server-stream.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "json.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <exception>
#include <functional>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <vector>

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    if (shutdown_handler) {
        shutdown_handler(signal);
    }
}

// upstream server.cpp's ex_wrapper: a handler never throws; an exception becomes an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const common_json_error & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// gc_engine.cpp's tool_call_markers (current CortexAgent: grammar triggers from the chat session),
// unchanged in substance: the loaded chat template's tool-call
// opening markers that are one special token of this vocabulary, as {"markers": [{text, token}]}.
static server_http_res_ptr tool_call_markers(server_context & ctx) {
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "probe"}}})},
        {"tools", json::array({{
            {"type", "function"},
            {"function", {
                {"name", "probe"},
                {"description", "probe"},
                {"parameters", {{"type", "object"}, {"properties", json::object()}}},
            }},
        }})},
        {"tool_choice", "auto"},
    };
    std::vector<raw_buffer> files;
    const server_context_meta meta = ctx.get_meta();
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx.get_llama_context()));
    // The chat session owns the template's grammar triggers; applying it to sampling settings
    // hands them over as the server's own requests receive them.
    common_chat_session session;
    oaicompat_chat_params_parse(vocab, body, meta.chat_params, files, session);
    common_params_sampling sampling;
    session.apply_sampling(sampling);
    json markers = json::array();
    std::vector<llama_token> seen;
    for (const auto & trigger : sampling.grammar_triggers) {
        if (trigger.type != COMMON_GRAMMAR_TRIGGER_TYPE_WORD && trigger.type != COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            continue;
        }
        const std::string text = trigger.value;
        const std::vector<llama_token> tokens = common_tokenize(vocab, text, false, true);
        if (tokens.size() != 1 || std::find(seen.begin(), seen.end(), tokens[0]) != seen.end()) {
            continue;
        }
        const int attributes = llama_vocab_get_attr(vocab, tokens[0]);
        if ((attributes & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) == 0
            || common_token_to_piece(vocab, tokens[0], true) != text) {
            continue;
        }
        seen.push_back(tokens[0]);
        markers.push_back({{"text", text}, {"token", tokens[0]}});
    }
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str({{"markers", markers}});
    return res;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    signal(SIGPIPE, SIG_IGN);

    common_params params;
    common_init();
    server_stream_session_manager_start();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        server_stream_session_manager_stop();
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "a model (--model) is required\n");
        server_stream_session_manager_stop();
        return 1;
    }
    // CortexModelServer patch: as upstream server.cpp's server_prepare_params for one model (the
    // alias /v1/models reports); gc_engine runs none of server_prepare_params.
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    SRV_INF("%s", "initializing ...\n");
    llama_backend_init();
    llama_numa_init(params.numa);
    common_params_print_info(params, true);

    server_context ctx_server;
    server_routes routes(params, ctx_server);
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    // gc_engine's routes (llama-server's paths for them)
    ctx_http.get ("/health",                           ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",                        ex_wrapper(routes.get_health));
    ctx_http.get ("/slots",                            ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",                   ex_wrapper(routes.post_slots));
    ctx_http.post("/v1/chat/completions",              ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/completion",                       ex_wrapper(routes.post_completions));
    ctx_http.post("/apply-template",                   ex_wrapper(routes.post_apply_template));
    ctx_http.post("/tokenize",                         ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",                       ex_wrapper(routes.post_detokenize));
    ctx_http.get ("/tool-call-markers",                ex_wrapper([&ctx_server](const server_http_req &) { return tool_call_markers(ctx_server); }));
    // CortexModelServer patch: read-only facts the plugin reads (native context) and the frame may
    // show; gc_engine has no such routes
    ctx_http.get ("/v1/models",                        ex_wrapper(routes.get_models));
    ctx_http.get ("/models",                           ex_wrapper(routes.get_models));
    ctx_http.get ("/props",                            ex_wrapper(routes.get_props));
    ctx_http.get ("/metrics",                          ex_wrapper(routes.get_metrics));

    auto clean_up = [&ctx_http, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        server_stream_session_manager_stop();
        ctx_http.stop();
        ctx_server.terminate();
        llama_backend_free();
    };

    // CortexModelServer patch: the HTTP listener (gc_engine has none). As upstream: listen before
    // loading, so /health answers 503 while the model loads
    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }
    if (!ctx_server.load_model(params)) {
        clean_up();
        ctx_http.join();
        SRV_ERR("%s", "exiting due to model loading error\n");
        return 1;
    }
    routes.update_meta(ctx_server);
    ctx_http.is_ready.store(true);
    SRV_INF("%s", "model loaded\n");
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
    }

    shutdown_handler = [&](int) {
        ctx_server.terminate();
    };
    struct sigaction action;
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    // blocks until terminate()
    ctx_server.start_loop();

    clean_up();
    ctx_http.join();
    // as upstream's instance mode at exit
    if (auto * ll_ctx = ctx_server.get_llama_context(); ll_ctx != nullptr) {
        common_memory_breakdown_print(ll_ctx);
    }
    common_log_flush(common_log_main());
    return 0;
}
