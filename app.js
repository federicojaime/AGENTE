/* ------------------------------------------------------------------
 *  app.js  —  Servidor Express optimizado para:
 *               1) /upload  → indexa un PDF al vector‑store local
 *               2) /chat    → responde usando RAG con OpenAI
 *               3) /status  → verificar estado del sistema
 * ------------------------------------------------------------------ */

import "dotenv/config";
import express   from "express";
import cors      from "cors";
import multer    from "multer";
import fs        from "fs";
import path      from "path";
import { ingestPDF }          from "./embed.js";
import { similaritySearch, verifyVectorStore }   from "./utils/vectorStore.js";
import { OpenAI }             from "openai";

/* ---------- config ---------- */
const upload = multer({ dest: "uploads/" });
const openai = new OpenAI();
const app    = express();
const manualPath = path.resolve("./manual.pdf");

// Variables de estado
let manualCargado = false;
let ultimaCarga = null;
let totalChunks = 0;

app.use(cors({
  origin: '*',  // Permite todas las solicitudes CORS (ajustar en producción)
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());

/* ---------- Función para verificar si ya hay un manual cargado ---------- */
async function verificarVectorStore() {
  try {
    const result = await verifyVectorStore();
    if (result && result.length > 0) {
      console.log(`✅ Vector store encontrado con ${result.length} fragmentos`);
      manualCargado = true;
      totalChunks = result.length;
      ultimaCarga = new Date().toISOString();
      return true;
    }
    return false;
  } catch (err) {
    console.error("Error al verificar vector store:", err);
    return false;
  }
}

/* ---------- Función para precargar manual.pdf ---------- */
async function precargarManual() {
  // Primero verificamos si ya existe un vector store
  const existente = await verificarVectorStore();
  if (existente) {
    return true;
  }

  try {
    // Verifica si el archivo existe
    if (fs.existsSync(manualPath)) {
      console.log("📚 Precargando manual.pdf...");
      const total = await ingestPDF(manualPath);
      console.log(`✅ Manual indexado con éxito: ${total} fragmentos`);
      manualCargado = true;
      totalChunks = total;
      ultimaCarga = new Date().toISOString();
      return true;
    } else {
      console.error("❌ Error: No se encontró el archivo manual.pdf");
      return false;
    }
  } catch (err) {
    console.error("❌ Error al precargar manual:", err.message);
    return false;
  }
}

/* ---------- 1) subir / indexar ---------- */
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    const total = await ingestPDF(req.file.path);         // indexa y guarda
    manualCargado = true;
    totalChunks = total;
    ultimaCarga = new Date().toISOString();
    
    res.json({ 
      ok: true, 
      chunks: total,
      mensaje: `¡Documento indexado correctamente! Se generaron ${total} fragmentos.` 
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ 
      ok: false, 
      error: err.message,
      mensaje: "Lo sentimos, hubo un problema al procesar el archivo." 
    });
  }
});

/* ---------- 2) chat RAG ---------- */
app.post("/chat", async (req, res) => {
  const { message, history = [] } = req.body;
  if (!message?.trim()) return res.status(400).json({ 
    error: "message vacío",
    mensaje: "Por favor, escribe un mensaje para poder ayudarte.",
    manualCargado
  });

  try {
    // Verificamos que el vector store esté cargado
    if (!manualCargado) {
      await verificarVectorStore();
    }

    /* 2.1 embedding de la consulta */
    const [{ embedding }] = (await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: message
    })).data;

    /* 2.2 recuperación de contexto (top‑4) */
    const contextChunks = await similaritySearch(embedding, 4);
    const context = contextChunks.join("\n---\n");

    /* 2.3 generación */
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content:
          "Eres un asistente amable y cordial de la empresa. Tu objetivo es ayudar a los usuarios "+
          "respondiendo sus preguntas de manera clara, ordenada y fácil de entender. "+
          "Usa un tono cálido y cercano en tus respuestas. "+
          "Formatea tus respuestas utilizando Markdown para mejorar la legibilidad: "+
          "- Usa **texto en negrita** para resaltar conceptos importantes o títulos "+
          "- Utiliza listas numeradas para pasos secuenciales "+
          "- Usa saltos de línea para separar párrafos y mejorar la legibilidad "+
          "- Si mencionas secciones o categorías, resáltalas con **negrita** "+
          "Basa tus respuestas ÚNICAMENTE en la información del manual proporcionado. "+
          "Si la información solicitada no se encuentra en el manual, responde amablemente: "+
          "\"Lo siento, no he podido encontrar esa información específica en nuestro manual. "+
          "¿Puedo ayudarte con algo más o prefieres que consulte con un especialista?\""
        },
        ...history.slice(-4), // Limitamos el historial para evitar tokens excesivos
        { role: "user", content: `Pregunta: "${message}"\n\nContexto:\n${context}` }
      ],
      temperature: 0.7,
      max_tokens: 500,
    });

    res.json({ 
      answer: completion.choices[0].message.content,
      contextUsado: contextChunks.length,
      manualCargado
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ 
      error: err.message,
      mensaje: "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, inténtalo de nuevo.",
      manualCargado
    });
  }
});

/* ---------- 3) verificar estado ---------- */
app.get("/status", async (req, res) => {
  // Verificamos de nuevo el vector store por si se cargó desde otra instancia
  if (!manualCargado) {
    await verificarVectorStore();
  }
  
  res.json({
    ok: true,
    manualCargado,
    ultimaCarga,
    chunks: totalChunks,
    version: "1.0.1"
  });
});

/* ---------- 4) arranque ---------- */
const PORT = process.env.PORT || 4000;

// Precargar el manual antes de iniciar el servidor
precargarManual().then(success => {
  app.listen(PORT, () => {
    console.log(`🚀 Backend listo en http://localhost:${PORT}`);
    if (success) {
      console.log(`📘 Manual precargado y listo para consultas (${totalChunks} fragmentos)`);
    } else {
      console.log("⚠️ El manual no se precargó correctamente. Sube el archivo manualmente.");
    }
  });
});