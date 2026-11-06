const express = require("express");
const cors = require("cors");

const app = express();

// ---------- Настройки ----------
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || "claude-sonnet-5-5";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || "*";

const PROMPT =
  "Реши задачу для 8 класса. Пиши просто, как в школе. Если не можешь — верни ошибку";

// ---------- Middleware ----------
app.use(cors({ origin: ALLOWED_ORIGIN }));
app.use(express.json({ limit: "15mb" })); // фото в base64 бывают тяжёлыми

// ---------- Вспомогательные функции ----------
class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// Убираем префикс "data:image/jpeg;base64," если он есть, и определяем тип картинки
function parseImage(input) {
  let data = input.trim();
  let mediaType = null;

  const match = data.match(/^data:(image\/[a-zA-Z+]+);base64,/);
  if (match) {
    mediaType = match[1];
    data = data.slice(match[0].length);
  }

  if (!mediaType) {
    if (data.startsWith("/9j/")) mediaType = "image/jpeg";
    else if (data.startsWith("iVBOR")) mediaType = "image/png";
    else if (data.startsWith("R0lGOD")) mediaType = "image/gif";
    else if (data.startsWith("UklGR")) mediaType = "image/webp";
    else mediaType = "image/jpeg"; // по умолчанию
  }

  return { data, mediaType };
}

// Нужно ли переключаться на Gemini?
function shouldFallback(err) {
  return err.status === 429 || err.status >= 500 || err.isNetworkError === true;
}

// ---------- Claude ----------
async function solveWithClaude(image) {
  if (!ANTHROPIC_API_KEY) {
    throw new ApiError("Не задан ANTHROPIC_API_KEY", 503);
  }
  const { data, mediaType } = parseImage(image);

  let response;
  try {
    response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: CLAUDE_MODEL,
        max_tokens: 2000,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image",
                source: { type: "base64", media_type: mediaType, data },
              },
              { type: "text", text: PROMPT },
            ],
          },
        ],
      }),
      signal: AbortSignal.timeout(60000),
    });
  } catch (e) {
    const err = new ApiError(`Claude недоступен: ${e.message}`, 0);
    err.isNetworkError = true;
    throw err;
  }

  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(`Claude вернул ошибку ${response.status}: ${text}`, response.status);
  }

  const json = await response.json();
  const text = (json.content || [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();

  if (!text) throw new ApiError("Claude вернул пустой ответ", 502);
  return text;
}

// ---------- Gemini ----------
async function solveWithGemini(image) {
  if (!GEMINI_API_KEY) {
    throw new ApiError("Не задан GEMINI_API_KEY", 503);
  }
  const { data, mediaType } = parseImage(image);

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: PROMPT },
              { inline_data: { mime_type: mediaType, data } },
            ],
          },
        ],
      }),
      signal: AbortSignal.timeout(60000),
    });
  } catch (e) {
    throw new ApiError(`Gemini недоступен: ${e.message}`, 502);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(`Gemini вернул ошибку ${response.status}: ${text}`, response.status);
  }

  const json = await response.json();
  const parts = json.candidates?.[0]?.content?.parts || [];
  const text = parts.map((p) => p.text || "").join("\n").trim();

  if (!text) throw new ApiError("Gemini вернул пустой ответ", 502);
  return text;
}

// ---------- Маршруты ----------
app.get("/", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/analyze", async (req, res, next) => {
  try {
    const { image } = req.body || {};
    if (!image || typeof image !== "string") {
      return res.status(400).json({ error: 'Нужно передать JSON вида { "image": "base64_string" }' });
    }

    let solution;
    try {
      solution = await solveWithClaude(image);
    } catch (claudeError) {
      if (shouldFallback(claudeError)) {
        console.warn("Claude не сработал, пробую Gemini:", claudeError.message);
        solution = await solveWithGemini(image);
      } else {
        throw claudeError;
      }
    }

    res.json({ solution });
  } catch (err) {
    next(err);
  }
});

// ---------- Обработка ошибок ----------
app.use((err, req, res, next) => {
  console.error("Ошибка:", err.message);

  // Слишком большой запрос или кривой JSON
  if (err.type === "entity.too.large") {
    return res.status(413).json({ error: "Фото слишком большое. Уменьши размер." });
  }
  if (err.type === "entity.parse.failed") {
    return res.status(400).json({ error: "Некорректный JSON" });
  }

  const status = err.status >= 400 && err.status < 600 ? err.status : 500;
  res.status(status).json({ error: err.message || "Внутренняя ошибка сервера" });
});

app.listen(PORT, () => {
  console.log(`Сервер запущен на порту ${PORT}`);
});
