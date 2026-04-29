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

        // 2. Load the model and run the tool-calling agent loop. The session
        //    is AutoCloseable — close it to release off-heap memory.
        try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath, /* systemPrompt */ null)) {
            JavaAgentLoop agent = JavaAgentLoop.builder()
                .session(session)
                .tool(calculator)
                .systemPrompt("You are a helpful assistant. When asked an arithmetic "
                    + "question, call the calculator tool exactly once.")
                .config(new AgentConfig())
                .template("llama3")  // works for Llama 3.x; auto-detects if omitted
                .metadata(new ModelMetadata())
                .build();

            // Stream tokens to stdout as they arrive, then print the final answer.
            String finalResponse = agent.chat(prompt, System.out::print);

            System.out.println();
            System.out.println("---");
            System.out.println("Final answer: " + finalResponse);
        }
    }
}
