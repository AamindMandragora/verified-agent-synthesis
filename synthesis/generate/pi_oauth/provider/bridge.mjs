import { createHash } from "node:crypto";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";

const PROVIDER = "openai-codex";
const MODEL = "gpt-5.6-sol";
const CLAIM_PATH = "https://api.openai.com/auth";
const COMPLETE_KEYS = new Set([
  "operation",
  "model",
  "system_prompt",
  "user_prompt",
  "reasoning",
  "tools",
  "tool_choice",
  "previous_response_id",
  "conversation",
]);

function emit(payload, exitCode = 0) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
  process.exitCode = exitCode;
}

function errorCategory(error) {
  const text = String(error instanceof Error ? error.message : error).toLowerCase();
  if (/auth|oauth|credential|token|login|subscription/.test(text)) return "authentication";
  if (/rate|quota|limit|credit/.test(text)) return "quota";
  if (/timeout|timed out|abort/.test(text)) return "timeout";
  if (/model|not found|unsupported/.test(text)) return "model_unavailable";
  return "provider";
}

function decodeAccountId(accessToken) {
  const parts = accessToken.split(".");
  if (parts.length !== 3) throw new Error("invalid OAuth access token");
  const payload = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
  const accountId = payload?.[CLAIM_PATH]?.chatgpt_account_id;
  if (typeof accountId !== "string" || accountId.length === 0) {
    throw new Error("OAuth account identity missing");
  }
  return accountId;
}

function validateCompleteRequest(request) {
  if (
    Object.keys(request).some((key) => !COMPLETE_KEYS.has(key)) ||
    request.operation !== "complete" ||
    request.model !== MODEL ||
    typeof request.system_prompt !== "string" ||
    request.system_prompt.length === 0 ||
    typeof request.user_prompt !== "string" ||
    request.user_prompt.length === 0 ||
    request.reasoning !== "high" ||
    !Array.isArray(request.tools) ||
    request.tools.length !== 0 ||
    request.tool_choice !== "none" ||
    request.previous_response_id !== null ||
    request.conversation !== null
  ) {
    throw new Error("invalid_request");
  }
}

async function runtimeAndRoute() {
  const authPath = process.env.CSD_PI_AUTH_PATH;
  if (!authPath) throw new Error("OAuth credential path missing");
  const runtime = await ModelRuntime.create({
    authPath,
    modelsPath: null,
    allowModelNetwork: false,
  });
  if (!runtime.isUsingOAuth(PROVIDER)) {
    throw new Error("ChatGPT/Codex OAuth is not configured");
  }
  const auth = await runtime.getAuth(PROVIDER, { minOAuthValidityMs: 300000 });
  const accessToken = auth?.auth?.apiKey;
  if (typeof accessToken !== "string" || accessToken.length === 0) {
    throw new Error("ChatGPT/Codex OAuth is unavailable");
  }
  const accountId = decodeAccountId(accessToken);
  return {
    runtime,
    route: {
      auth_mode: "chatgpt_codex_oauth",
      provider: PROVIDER,
      model: MODEL,
      account_id_sha256: createHash("sha256").update(accountId).digest("hex"),
    },
  };
}

async function readRequest() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  if (chunks.reduce((total, chunk) => total + chunk.length, 0) > 16 * 1024 * 1024) {
    throw new Error("invalid_request");
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

try {
  const request = await readRequest();
  if (
    typeof request !== "object" ||
    request === null ||
    Array.isArray(request) ||
    !["check_auth", "complete"].includes(request.operation)
  ) {
    emit({ ok: false, error_category: "invalid_request" }, 2);
  } else if (request.operation === "check_auth") {
    if (Object.keys(request).length !== 1) {
      emit({ ok: false, error_category: "invalid_request" }, 2);
    } else {
      const { route } = await runtimeAndRoute();
      emit({ ok: true, route });
    }
  } else {
    validateCompleteRequest(request);
    const { runtime, route } = await runtimeAndRoute();
    const model = runtime.getModel(PROVIDER, MODEL);
    if (!model) throw new Error("requested model is unavailable");
    const response = await runtime.completeSimple(
      model,
      {
        systemPrompt: request.system_prompt,
        messages: [
          {
            role: "user",
            content: request.user_prompt,
            timestamp: Date.now(),
          },
        ],
        tools: [],
      },
      { reasoning: "high", toolChoice: "none" },
    );
    const toolCalls = response.content.filter((item) => item.type === "toolCall");
    const text = response.content
      .filter((item) => item.type === "text")
      .map((item) => item.text)
      .join("")
      .trim();
    if (response.stopReason === "error" || response.stopReason === "aborted") {
      throw new Error(errorCategory(response.errorMessage));
    }
    if (toolCalls.length > 0 || text.length === 0) {
      throw new Error("provider returned no plain text answer");
    }
    emit({ ok: true, text, route });
  }
} catch (error) {
  const category = String(error instanceof Error ? error.message : error) === "invalid_request"
    ? "invalid_request"
    : errorCategory(error);
  emit({ ok: false, error_category: category }, 3);
}
