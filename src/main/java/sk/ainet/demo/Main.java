package sk.ainet.demo;

import sk.ainet.apps.kllama.chat.AgentConfig;
import sk.ainet.apps.kllama.chat.ModelMetadata;
import sk.ainet.apps.kllama.chat.ToolDefinition;
import sk.ainet.apps.kllama.chat.java.JavaAgentLoop;
import sk.ainet.apps.kllama.chat.java.JavaTool;
import sk.ainet.apps.kllama.chat.java.JavaTools;
import sk.ainet.apps.kllama.java.KLlamaJava;
import sk.ainet.apps.kllama.java.KLlamaSession;

import java.nio.file.Path;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Minimal Llama tool-calling demo in pure Java.
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -Dexec.args="path/to/model.gguf 'What is 17 * 23?'"
 * </pre>
 *
 * <p>Or after `mvn package`:
 * <pre>
 *   java --enable-preview --add-modules jdk.incubator.vector \
 *        -jar target/skainet-java-demo-0.1.0-SNAPSHOT-shaded.jar \
 *        path/to/model.gguf 'What is 17 * 23?'
 * </pre>
 */
public final class Main {

    private Main() {
        // utility entry point
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println(
                "Usage: skainet-java-demo <model.gguf> [prompt]\n"
                + "Example:\n"
                + "  skainet-java-demo Llama-3.2-1B-Instruct-Q8_0.gguf 'What is 17 * 23?'"
            );
            System.exit(1);
        }

        Path modelPath = Path.of(args[0]);
        String prompt = args.length >= 2 ? args[1] : "What is 17 * 23?";

        String systemPrompt = "You are a helpful assistant. When asked an arithmetic "
            + "question, call the calculator tool exactly once.";
        String templateName = "llama3";  // works for Llama 3.x; auto-detects if omitted

        // Track whether the assistant is currently streaming a token line, so
        // [TOOL CALL] / [TOOL RESULT] logs can break the line cleanly mid-stream.
        final boolean[] assistantStreaming = {false};

        // 1. Define a calculator tool the model can call. The arguments map
        //    is decoded from the model's JSON output as plain Java types
        //    (String, Long, Double, Boolean, List, nested Map).
        JavaTool calculator = new JavaTool() {
            @Override
            public ToolDefinition getDefinition() {
                return JavaTools.definition(
                    "calculator",
                    "Evaluate a simple arithmetic expression like '17 * 23'.",
                    "{\"type\":\"object\","
                        + "\"properties\":{\"expression\":{\"type\":\"string\","
                        + "\"description\":\"Arithmetic expression to evaluate\"}},"
                        + "\"required\":[\"expression\"]}"
                );
            }

            @Override
            public String execute(Map<String, ?> arguments) {
                Object exprObj = arguments.get("expression");
                if (exprObj == null) return "error: missing expression";
                String s = exprObj.toString().trim();
                String[] parts = s.split("\\s*\\*\\s*");
                if (parts.length != 2) {
                    return "error: only 'a * b' is supported in this minimal demo";
                }
                try {
                    long a = Long.parseLong(parts[0]);
                    long b = Long.parseLong(parts[1]);
                    return String.valueOf(a * b);
                } catch (NumberFormatException e) {
                    return "error: " + e.getMessage();
                }
            }
        };

        // Wrap each tool with a logging decorator so we see what the model
        // actually called, with which arguments, and what we returned. The
        // JavaAgentLoop dispatches tool calls through JavaTool.execute(), so
        // this is the cleanest hook the Java surface exposes.
        JavaTool loggedCalculator = loggingTool(calculator, assistantStreaming);

        // Up-front: dump the configuration the agent will run with — model,
        // chat template, system prompt, and the tool catalogue the model is
        // told it can call (name + description + JSON-Schema parameters).
        System.out.println("[CONFIG]    model=" + modelPath);
        System.out.println("[CONFIG]    template=" + templateName);
        System.out.println("[SYSTEM]    " + systemPrompt);
        ToolDefinition def = calculator.getDefinition();
        System.out.println("[TOOLS]     1 tool registered:");
        System.out.println("            - " + def.getName() + ": " + def.getDescription());
        System.out.println("              schema: " + def.getParameters());
        System.out.println("[USER]      " + prompt);

        // 2. Load the model and run the tool-calling agent loop. The session
        //    is AutoCloseable — close it to release off-heap memory.
        try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath, /* systemPrompt */ null)) {
            JavaAgentLoop agent = JavaAgentLoop.builder()
                .session(session)
                .tool(loggedCalculator)
                .systemPrompt(systemPrompt)
                .config(new AgentConfig())
                .template(templateName)
                .metadata(new ModelMetadata())
                .build();

            // Stream tokens to stdout as they arrive, while measuring throughput.
            // The Consumer<String> callback fires once per decoded token, so
            // counting calls = counting tokens. Wall-clock includes prefill
            // ("time to first token"); decode-only excludes it.
            final long[] firstTokenNanos = {0};
            final long[] lastTokenNanos = {0};
            final int[] tokenCount = {0};

            Consumer<String> meter = token -> {
                long now = System.nanoTime();
                if (firstTokenNanos[0] == 0) firstTokenNanos[0] = now;
                lastTokenNanos[0] = now;
                tokenCount[0]++;
                if (!assistantStreaming[0]) {
                    System.out.print("[ASSISTANT] ");
                    assistantStreaming[0] = true;
                }
                System.out.print(token);
            };

            long startWallNanos = System.nanoTime();
            String finalResponse = agent.chat(prompt, meter);
            long endWallNanos = System.nanoTime();
            if (assistantStreaming[0]) {
                System.out.println();
                assistantStreaming[0] = false;
            }

            int n = tokenCount[0];
            double wallSec = (endWallNanos - startWallNanos) / 1e9;
            double decodeSec = (lastTokenNanos[0] - firstTokenNanos[0]) / 1e9;
            double wallTps = wallSec > 0 ? n / wallSec : 0;
            double decodeTps = (n > 1 && decodeSec > 0) ? (n - 1) / decodeSec : 0;

            System.out.println("---");
            System.out.println("[FINAL]     " + finalResponse);
            System.out.printf(
                "[STATS]     %d tokens — wall %.2fs (%.2f tok/s), decode %.2fs (%.2f tok/s)%n",
                n, wallSec, wallTps, decodeSec, decodeTps);
        }
    }

    /**
     * Wraps a {@link JavaTool} so each invocation prints the call site
     * (tool name + decoded arguments) and the returned value. The
     * {@code streamingFlag} is the shared "assistant is mid-stream" latch
     * from {@link #main} — when set, we emit a leading newline so the log
     * line doesn't get appended to a half-finished assistant token line.
     */
    private static JavaTool loggingTool(JavaTool delegate, boolean[] streamingFlag) {
        return new JavaTool() {
            @Override
            public ToolDefinition getDefinition() {
                return delegate.getDefinition();
            }

            @Override
            public String execute(Map<String, ?> arguments) {
                if (streamingFlag[0]) {
                    System.out.println();
                    streamingFlag[0] = false;
                }
                String name = delegate.getDefinition().getName();
                System.out.println("[TOOL CALL] " + name + " args=" + arguments);
                String result = delegate.execute(arguments);
                System.out.println("[TOOL RESULT] " + name + " -> " + result);
                return result;
            }
        };
    }
}
