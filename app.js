/* ------------------------------------------------------------------
 *  app.js  —  Servidor Express optimizado para:
 *               1) /upload  → indexa un PDF al vector‑store local
 *               2) /chat    → responde usando RAG con OpenAI
 *               3) /status  → verificar estado del sistema
 *               4) /manuals → listar manuales disponibles
 * ------------------------------------------------------------------ */

import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from 'url';
import { ingestPDF } from "./embed.js";
import { similaritySearch, verifyVectorStore, getAvailableManuals } from "./utils/vectorStore.js";
import { OpenAI } from "openai";

/* ---------- config ---------- */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const upload = multer({ dest: "uploads/" });
const openai = new OpenAI();
const app = express();
const MANUALES_DIR = path.resolve(__dirname, "./manuales");  // directorio para los PDFs precargados

// Variables de estado
let manualesCargados = false;
let ultimaCarga = null;
let totalChunks = 0;
let manualesInfo = [];

app.use(cors({
  origin: '*',  // Permite todas las solicitudes CORS (ajustar en producción)
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());
app.use('/manuals', express.static(path.join(__dirname, 'public/manuals'))); // Servir PDFs

/* ---------- Función para verificar si ya hay manuales cargados ---------- */
async function verificarVectorStore() {
  try {
    const result = await verifyVectorStore();
    if (result && result.length > 0) {
      console.log(`✅ Vector store encontrado con ${result.length} fragmentos`);
      
      // Obtener información de los manuales disponibles
      manualesInfo = await getAvailableManuals();
      manualesCargados = true;
      totalChunks = result.length;
      ultimaCarga = new Date().toISOString();
      
      console.log(`📚 Manuales cargados: ${manualesInfo.length}`);
      return true;
    }
    return false;
  } catch (err) {
    console.error("Error al verificar vector store:", err);
    return false;
  }
}

/* ---------- Función para precargar todos los manuales ---------- */
async function precargarManuales() {
  // Primero verificamos si ya existen manuales en el vector store
  const existente = await verificarVectorStore();
  if (existente) {
    return true;
  }

  try {
    // Creamos el directorio de manuales si no existe
    if (!fs.existsSync(MANUALES_DIR)) {
      console.log(`📁 Creando directorio de manuales: ${MANUALES_DIR}`);
      fs.mkdirSync(MANUALES_DIR, { recursive: true });
      console.log("❌ No se encontraron manuales para precargar");
      return false;
    }

    // Leemos todos los PDFs del directorio
    const archivos = fs.readdirSync(MANUALES_DIR).filter(file => file.toLowerCase().endsWith('.pdf'));
    
    if (archivos.length === 0) {
      console.log("❌ No se encontraron PDFs en el directorio de manuales");
      return false;
    }
    
    console.log(`📚 Precargando ${archivos.length} manuales...`);
    
    // Procesamos cada PDF
    for (const archivo of archivos) {
      const rutaArchivo = path.join(MANUALES_DIR, archivo);
      console.log(`   📄 Procesando: ${archivo}...`);
      
      try {
        const resultado = await ingestPDF(rutaArchivo, archivo);
        totalChunks += resultado.chunks;
        manualesInfo.push(resultado);
        console.log(`   ✅ Manual indexado: ${archivo} (${resultado.chunks} fragmentos)`);
      } catch (err) {
        console.error(`   ❌ Error al procesar ${archivo}:`, err.message);
      }
    }
    
    manualesCargados = manualesInfo.length > 0;
    ultimaCarga = new Date().toISOString();
    
    console.log(`✅ Proceso de precarga finalizado. ${manualesInfo.length} manuales indexados con ${totalChunks} fragmentos totales.`);
    return manualesCargados;
  } catch (err) {
    console.error("❌ Error al precargar manuales:", err.message);
    return false;
  }
}

/* ---------- 1) subir / indexar (opcional, para agregar más manuales) ---------- */
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    const resultado = await ingestPDF(req.file.path, req.file.originalname);
    manualesCargados = true;
    totalChunks += resultado.chunks;
    ultimaCarga = new Date().toISOString();
    manualesInfo.push(resultado);
    
    res.json({ 
      ok: true, 
      manual: resultado,
      mensaje: `¡Documento indexado correctamente! Se generaron ${resultado.chunks} fragmentos.` 
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
    manualesCargados
  });

  try {
    // Verificamos que el vector store esté cargado
    if (!manualesCargados) {
      const verificado = await verificarVectorStore();
      if (!verificado) {
        return res.status(400).json({
          error: "manuales_no_cargados",
          mensaje: "No hay manuales cargados en el sistema. Por favor, precarga los manuales primero.",
          manualesCargados: false
        });
      }
    }

    /* 2.1 embedding de la consulta */
    const [{ embedding }] = (await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: message
    })).data;

    /* 2.2 recuperación de contexto (top‑4) */
    const contextResults = await similaritySearch(embedding, 4);
    if (contextResults.length === 0) {
      return res.json({
        answer: "Lo siento, no he podido encontrar información relevante en nuestros manuales. ¿Podrías reformular tu pregunta o intentar con otro tema?",
        manualesCargados
      });
    }

    // Extraer los textos y la información del manual más relevante
    const contextTexts = contextResults.map(result => result.text);
    const context = contextTexts.join("\n---\n");

    // Obtener el manual más relevante (el que tiene más coincidencias en los resultados)
    const manualCounts = {};
    let topManual = null;
    let topCount = 0;

    for (const result of contextResults) {
      manualCounts[result.manualId] = (manualCounts[result.manualId] || 0) + 1;
      if (manualCounts[result.manualId] > topCount) {
        topCount = manualCounts[result.manualId];
        topManual = {
          id: result.manualId,
          info: result.manualInfo
        };
      }
    }

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
          "¿Puedo ayudarte con algo más o prefieres que consulte con un especialista?\"" +
          "Al final de tu respuesta, incluye un enlace al manual consultado con el texto: 'Ver manual: [título del manual]'"
        },
        ...history.slice(-4), // Limitamos el historial para evitar tokens excesivos
        { role: "user", content: `Pregunta: "${message}"\n\nContexto:\n${context}` }
      ],
      temperature: 0.7,
      max_tokens: 500,
    });

    let answer = completion.choices[0].message.content;
    
    // Agregar enlace al manual si no lo tiene ya
    if (topManual && !answer.includes("Ver manual:")) {
      answer += `\n\n---\n[Ver manual: ${topManual.info.title}](/manuals/${path.basename(topManual.info.path)})`;
    }

    res.json({ 
      answer,
      manualInfo: topManual,
      contextUsado: contextResults.length,
      manualesCargados
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ 
      error: err.message,
      mensaje: "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, inténtalo de nuevo.",
      manualesCargados
    });
  }
});

/* ---------- 3) verificar estado ---------- */
app.get("/status", async (req, res) => {
  // Verificamos de nuevo el vector store por si se cargó desde otra instancia
  if (!manualesCargados) {
    await verificarVectorStore();
  }
  
  res.json({
    ok: true,
    manualesCargados,
    ultimaCarga,
    manuales: manualesInfo.length,
    chunks: totalChunks,
    version: "1.1.0"
  });
});

/* ---------- 4) listar manuales ---------- */
app.get("/manuals", async (req, res) => {
  try {
    // Aseguramos que tenemos la información actualizada
    if (!manualesCargados) {
      await verificarVectorStore();
    }
    
    // Si aún no tenemos manuales, intentamos obtenerlos
    if (manualesInfo.length === 0) {
      manualesInfo = await getAvailableManuals();
    }
    
    res.json({
      ok: true,
      manuales: manualesInfo
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      ok: false,
      error: err.message,
      mensaje: "Error al obtener la lista de manuales."
    });
  }
});

/* ---------- 5) arranque ---------- */
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`🚀 Servidor ejecutándose en http://localhost:${PORT}`);
});
// Precargar los manuales antes de iniciar el servidor
precargarManuales().then(success => {
  app.listen(PORT, () => {
    console.log(`🚀 Backend listo en http://localhost:${PORT}`);
    if (success) {
      console.log(`📘 ${manualesInfo.length} manuales precargados y listos para consultas (${totalChunks} fragmentos totales)`);
    } else {
      console.log("⚠️ No se precargaron manuales correctamente. Verifica el directorio de manuales.");
    }
  });
});