// ============================================================
//  School Helper — сервер (Node.js + Express + Gemini)
//
//  Переменные окружения (Render → Environment):
//    GEMINI_API_KEY          — ключ Gemini (обязательно)
//    GEMINI_MODEL            — основная модель (по умолчанию gemini-3.8-flash)
//    GEMINI_FALLBACK_MODEL   — запасная модель на случай перегрузки основной
//                              (по умолчанию gemini-3.5-flash-lite; пустое значение — отключить)
//    PORT                    — порт (Render задаёт сам)
//
//  Нужен Node.js 18 или новее (используется встроенный fetch).
// ============================================================

const express = require("express");
const cors = require("cors");

const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;
const MODEL = process.env.GEMINI_MODEL || "gemini-3.8-flash";
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL ?? "gemini-3.5-flash-lite";
const GEMINI_TIMEOUT_MS = 60000;      // сколько ждать ответ Gemini на один запрос
const MAX_PROMPT_CHARS = 2000;
const MAX_SOLUTION_CHARS = 20000;

// ---------- Промпты ----------

// Сценарий 1: новая задача
const SOLVE_SYSTEM_PROMPT =
  "Ты — школьный учитель. Реши задачу для 8-9 класса. " +
  "Оформи ответ строго текстом ТАК, КАК УЧЕНИК ДОЛЖЕН ЗАПИСАТЬ ЕГО В ТЕТРАДЬ. " +
  "Используй понятные текстовые формулы. " +
  "НИКАКИХ рисунков, схем или ASCII-графики. Только текст, шаги решения и ответ.";

// Сценарий 2: действие над готовым ответом
const ACTION_SYSTEM_PROMPT =
  "Ты — школьный учитель (8-9 класс). Отвечай только текстом, понятными текстовыми формулами. " +
  "НИКАКИХ рисунков, схем или ASCII-графики.";

const ACTION_PROMPTS = {
  explain: "Объясни это решение по шагам, простыми словами, как будто я ничего не понимаю.",
  shorten: "Сократи это решение. Оставь только самое главное и ответ.",
  rephrase: "Перефразируй это решение другими словами, сохранив смысл."
};

// Приложение сейчас присылает русские названия кнопок — понимаем и их, и английские ключи
const ACTION_ALIASES = {
  explain: "explain", "объяснить": "explain",
  shorten: "shorten", "короче": "shorten",
  rephrase: "rephrase", "перефразировать": "rephrase"
};

// ---------- Ошибки ----------

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// Эти статусы Gemini означают «попробуй другую модель»: перегрузка, лимит, нет такой модели, сбой
const FALLBACK_STATUSES = [404, 429, 500, 502, 503, 504];

// ---------- Сервер ----------

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Проверка «жив ли сервер» (можно открыть в браузере, чтобы разбудить Render)
app.get("/", (req, res) => res.send("School Helper server is running"));
app.get("/health", (req, res) => res.json({ ok: true, model: MODEL }));

// ---------- Подготовка запроса ----------

function cleanString(value, maxLen) {
  return typeof value === "string" ? value.trim().slice(0, maxLen) : "";
}

// Приложение шлёт «чистую» base64, но на всякий случай убираем префикс data:...
function parseImage(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  const match = value.trim().match(/^data:(image\/[\w.+-]+);base64,(.*)$/s);
  if (match) return { mimeType: match[1], data: match[2] };
  return { mimeType: "image/jpeg", data: value.trim() };
}

function buildSolveRequest(body) {
  // приложение шлёт поле text, в ТЗ оно называется prompt — принимаем оба
  const prompt = cleanString(body.prompt ?? body.text, MAX_PROMPT_CHARS);
  const image = parseImage(body.image);

  if (!image && !prompt) {
    throw new HttpError(400, "Нужно прислать фото (image) или текст задания (prompt)");
  }

  const parts = [];
  if (image) {
    parts.push({ inlineData: { mimeType: image.mimeType, data: image.data } });
    parts.push({
      text: prompt
        ? "Реши задание на фото. Уточнение ученика: " + prompt
        : "Реши задание на фото."
    });
  } else {
    parts.push({ text: prompt });
  }

  return { systemPrompt: SOLVE_SYSTEM_PROMPT, parts };
}

function buildActionRequest(body) {
  const key = ACTION_ALIASES[String(body.action).trim().toLowerCase()];
  if (!key) {
    throw new HttpError(400, "Неизвестное действие. Допустимо: explain, shorten, rephrase");
  }

  const previous = cleanString(body.previousSolution, MAX_SOLUTION_CHARS);
  if (!previous) {
    throw new HttpError(400, "Для действия нужен текст предыдущего решения (previousSolution)");
  }

  // Картинка здесь не нужна — отправляем только текст
  return {
    systemPrompt: ACTION_SYSTEM_PROMPT,
    parts: [{ text: ACTION_PROMPTS[key] + "\n\nРешение:\n" + previous }]
  };
}

// ---------- Запрос к Gemini ----------

function extractText(data) {
  const candidate = data.candidates && data.candidates[0];
  const parts = (candidate && candidate.content && candidate.content.parts) || [];
  const text = parts
    .filter((p) => typeof p.text === "string" && !p.thought)
    .map((p) => p.text)
    .join("")
    .trim();
  if (text) return text;

  const blockReason = data.promptFeedback && data.promptFeedback.blockReason;
  if (blockReason) throw new HttpError(422, "Gemini отклонил запрос (" + blockReason + ")");

  const finish = candidate && candidate.finishReason;
  throw new HttpError(
    finish === "SAFETY" ? 422 : 502,
    "Gemini вернул пустой ответ" + (finish ? " (" + finish + ")" : "")
  );
}

async function callGemini(model, request) {
  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/" +
    encodeURIComponent(model) +
    ":generateContent";

  const payload = {
    systemInstruction: { parts: [{ text: request.systemPrompt }] },
    contents: [{ role: "user", parts: request.parts }]
  };

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": API_KEY },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(GEMINI_TIMEOUT_MS)
    });
  } catch (err) {
    const timedOut = err.name === "TimeoutError" || err.name === "AbortError";
    throw new HttpError(
      timedOut ? 504 : 502,
      timedOut ? "Gemini не ответил вовремя" : "Не удалось связаться с Gemini: " + err.message
    );
  }

  const raw = await response.text();
  if (!response.ok) {
    throw new HttpError(response.status, "Gemini вернул ошибку " + response.status + ": " + raw.slice(0, 500));
  }

  let data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    throw new HttpError(502, "Gemini вернул неразборчивый ответ");
  }
  return extractText(data);
}

// Сначала основная модель; если она перегружена или недоступна — запасная
async function askGemini(request) {
  const models = [MODEL];
  if (FALLBACK_MODEL && FALLBACK_MODEL !== MODEL) models.push(FALLBACK_MODEL);

  let lastError;
  for (let i = 0; i < models.length; i++) {
    try {
      const text = await callGemini(models[i], request);
      if (i > 0) console.log("Ответ получен от запасной модели " + models[i]);
      return text;
    } catch (err) {
      lastError = err;
      const canFallback = FALLBACK_STATUSES.includes(err.status) && i < models.length - 1;
      console.error("Модель " + models[i] + ": " + err.message);
      if (!canFallback) break;
      console.log("Пробую запасную модель " + models[i + 1] + "...");
    }
  }
  throw lastError;
}

// ---------- POST /analyze ----------

app.post("/analyze", async (req, res) => {
  try {
    if (!API_KEY) throw new HttpError(500, "На сервере не задан GEMINI_API_KEY");

    const body = req.body || {};
    const hasAction = body.action !== undefined && body.action !== null && String(body.action).trim() !== "";

    // Сценарий 2: действие над готовым ответом (без картинки)
    // Сценарий 1: новая задача (фото и/или текст)
    const request = hasAction ? buildActionRequest(body) : buildSolveRequest(body);

    console.log(
      "Запрос: " + (hasAction ? "действие " + body.action : "новая задача") +
      (body.image ? ", есть фото" : "")
    );

    const solution = await askGemini(request);
    res.json({ solution });
  } catch (err) {
    const known = err instanceof HttpError;
    const status = known && err.status >= 400 && err.status <= 599 ? err.status : known ? 502 : 500;
    if (!known) console.error("Непредвиденная ошибка:", err);
    res.status(status).json({ error: known ? err.message : "Внутренняя ошибка сервера" });
  }
});

// Ошибки разбора JSON и слишком большого тела запроса
app.use((err, req, res, next) => {
  if (err && err.type === "entity.too.large") {
    return res.status(413).json({ error: "Фото слишком большое" });
  }
  if (err && err.type === "entity.parse.failed") {
    return res.status(400).json({ error: "Некорректный JSON в запросе" });
  }
  console.error("Ошибка сервера:", err);
  res.status(500).json({ error: "Внутренняя ошибка сервера" });
});

app.listen(PORT, () => {
  console.log("Сервер запущен на порту " + PORT + ", модель: " + MODEL +
    (FALLBACK_MODEL ? ", запасная: " + FALLBACK_MODEL : ""));
  if (!API_KEY) console.error("ВНИМАНИЕ: переменная GEMINI_API_KEY не задана!");
});
