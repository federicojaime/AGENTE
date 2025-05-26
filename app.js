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
import { ingestPDF, ingestRemotePDF } from "./embed.js";
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

/* ---------- Configuración de manuales remotos ---------- */
let REMOTE_MANUALS = [];
try {
  if (process.env.REMOTE_MANUALS) {
    let rawValue = process.env.REMOTE_MANUALS;

    // Eliminar el signo '=' u otros caracteres no JSON al inicio si existen
    while (rawValue.charAt(0) !== '[' && rawValue.length > 0) {
      rawValue = rawValue.substring(1);
    }

    // Intentar parsear
    try {
      REMOTE_MANUALS = JSON.parse(rawValue);
      console.log("Manuales remotos cargados correctamente:", REMOTE_MANUALS.length);
    } catch (parseError) {
      console.error("Error al parsear JSON de REMOTE_MANUALS:", parseError.message);

      // Plan B: Usar las URLs directamente si el JSON falla
      REMOTE_MANUALS = [
        { "url": "https://codeo.site/manual-crm/manual-induccion.pdf", "title": "Manual de Inducción" },
        { "url": "https://codeo.site/manual-crm/plan-de-negocios.pdf", "title": "Plan de Negocios" },
        { "url": "https://codeo.site/manual-crm/manual-ventas.pdf", "title": "Manual de Ventas" },
        { "url": "https://codeo.site/manual-crm/manual-del-distribuidor.pdf", "title": "Manual del Distribuidor" },
        { "url": "https://codeo.site/manual-crm/manual-de-financiamiento.pdf", "title": "Manual de Financiamiento" },
        { "url": "https://codeo.site/manual-crm/dossier.pdf", "title": "Dossier" }
      ];
      console.log("Usando lista de manuales predeterminada:", REMOTE_MANUALS.length);
    }
  }
} catch (error) {
  console.error("Error general al procesar REMOTE_MANUALS:", error.message);
  console.log("Valor recibido:", process.env.REMOTE_MANUALS);

  // Usar una configuración de respaldo en caso de cualquier error
  REMOTE_MANUALS = [
    { "url": "https://codeo.site/manual-crm/manual-induccion.pdf", "title": "Manual de Inducción" },
    { "url": "https://codeo.site/manual-crm/plan-de-negocios.pdf", "title": "Plan de Negocios" },
    { "url": "https://codeo.site/manual-crm/manual-ventas.pdf", "title": "Manual de Ventas" },
    { "url": "https://codeo.site/manual-crm/manual-del-distribuidor.pdf", "title": "Manual del Distribuidor" },
    { "url": "https://codeo.site/manual-crm/manual-de-financiamiento.pdf", "title": "Manual de Financiamiento" },
    { "url": "https://codeo.site/manual-crm/dossier.pdf", "title": "Dossier" }
  ];
  console.log("Usando configuración de respaldo:", REMOTE_MANUALS.length);
}

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

/* ---------- Función para precargar manuales ---------- */
async function precargarManuales() {
  // Primero verificamos si ya existen manuales en el vector store
  const existente = await verificarVectorStore();
  if (existente) {
    return true;
  }

  try {
    // Verificar si hay manuales locales
    let tieneManualLocal = false;

    if (fs.existsSync(MANUALES_DIR)) {
      const archivos = fs.readdirSync(MANUALES_DIR).filter(file => file.toLowerCase().endsWith('.pdf'));
      if (archivos.length > 0) {
        tieneManualLocal = true;
        console.log(`📚 Encontrados ${archivos.length} manuales locales para precargar...`);

        // Procesar manuales locales...
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
      }
    } else {
      console.log("📁 No se encontró el directorio de manuales locales");
    }

    // Si no hay manuales locales o además queremos cargar remotos, procesamos URLs
    if ((!tieneManualLocal || REMOTE_MANUALS.length > 0) && REMOTE_MANUALS.length > 0) {
      console.log(`📚 Cargando ${REMOTE_MANUALS.length} manuales remotos...`);

      for (const manual of REMOTE_MANUALS) {
        console.log(`   📄 Procesando manual remoto: ${manual.title || manual.url}...`);
        try {
          const resultado = await ingestRemotePDF(manual.url, manual.title, manual);
          totalChunks += resultado.chunks;
          manualesInfo.push(resultado);
          console.log(`   ✅ Manual remoto indexado: ${manual.title || manual.url} (${resultado.chunks} fragmentos)`);
        } catch (err) {
          console.error(`   ❌ Error al procesar manual remoto ${manual.title || manual.url}:`, err.message);
        }
      }
    } else if (!tieneManualLocal) {
      console.log("⚠️ No hay manuales locales ni remotos configurados");
    }

    manualesCargados = manualesInfo.length > 0;
    ultimaCarga = new Date().toISOString();

    if (manualesCargados) {
      console.log(`✅ Proceso de precarga finalizado. ${manualesInfo.length} manuales indexados con ${totalChunks} fragmentos totales.`);
    } else {
      console.log("❌ No se pudieron cargar manuales ni locales ni remotos.");
    }

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
  console.log("📝 Recibida consulta:", req.body.message);

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
    console.log(`🔍 Resultados encontrados: ${contextResults.length}`);

    if (contextResults.length > 0) {
      console.log(`   Mejor coincidencia (score): ${contextResults[0].score.toFixed(4)}`);
    }

    if (contextResults.length === 0) {
      console.log("⚠️ No se encontraron resultados relevantes para la consulta");
      return res.json({
        answer: "Lo siento, no he podido encontrar información relevante en nuestros manuales. ¿Podrías reformular tu pregunta o intentar con otro tema? Si necesitas ayuda adicional, puedes contactar directamente con un asesor: [Contactar por WhatsApp](https://wa.me/2657218215?text=Hola%2C%20necesito%20ayuda%20con%20una%20consulta%20sobre%20los%20manuales)",
        manualesCargados
      });
    }

    // Extraer los textos y la información del manual más relevante
    const contextTexts = contextResults.map(result => result.text);
    const context = contextTexts.join("\n---\n");
    console.log(`📚 Contexto recuperado (fragmentos): ${contextTexts.length}`);

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
          info: result.manualInfo,
          chunkIds: contextResults
            .filter(r => r.manualId === result.manualId)
            .map(r => r.chunkId || 'sin-id')
        };
      }
    }

    if (topManual) {
      console.log(`📕 Manual más relevante: ${topManual.info.title} (${topCount} coincidencias)`);
    }

    /* 2.3 generación */
    console.log("🤖 Generando respuesta con GPT...");
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `Eres un asistente educativo especializado en la formación de una fuerza de ventas. Estás diseñado para responder preguntas, enseñar procedimientos y asesorar en base a manuales oficiales de una empresa de venta directa.

**ESTILO DE COMUNICACIÓN:**
- Usa un tono cálido, profesional y cercano
- Sé paciente y motivador, incluso cuando debas repetir o aclarar conceptos
- Mantén siempre una actitud didáctica y comprensiva

**INSTRUCCIONES DE RESPUESTA:**
1. Responde ÚNICAMENTE con información contenida en los manuales proporcionados
2. Indica la fuente exacta de la información: documento y sección/página específica
3. Usa formato Markdown para mejorar la legibilidad:
   - **Negrita** para conceptos importantes y títulos
   - Listas numeradas para procedimientos secuenciales
   - Listas con viñetas para elementos relacionados
   - Saltos de línea para separar párrafos
4. Incluye ejemplos concretos cuando sea posible
5. Si la información no está en los documentos, responde exactamente: "Lo siento, no he podido encontrar esa información específica en nuestros manuales."

**GUÍA DE BÚSQUEDA SEGÚN PALABRAS CLAVE:**

**Manual de Financiamiento:**
Palabras clave: orden, crédito, financiamiento, límite de crédito, verificación, verify, análisis, cuenta por cobrar, morosidad, cobranza, procesamiento de órdenes, flujo de trabajo, programa de financiamiento del distribuidor (DFP), cesiones, página del distribuidor, reserva

**Revista Plan de Negocios ARGENTINA:**
Palabras clave: número de distribución, distribuidor junior, ascenso, emprendedor, compañía, network, bono, blue network, royal network, premier, master, elite, nivel de precios, ganancias

**Manual de Inducción VL 2025 R:**
Palabras clave: venta directa, historia de Royal Prestige, línea histórica, arena competitiva, proceso comercial, telemarketing, prospecto, embudo, ticket, referidos, 4 en 14, cita instantánea, demo, demostración, valor, objeciones, cierre de venta, negociación, estructura demostración, rompiendo el hielo, cocinar, garantía, carpeta, degustación, catálogo, encuesta, láminas, técnicas de venta, señales de compra, post venta, servicio, procedimientos, sendero de relacionamiento, SPV, órdenes de compra, financiado por Hy Cite, contado, financiamiento francés, cotizador, Docucite, formulario, comisión por venta, sistema de cocina, acero 316L, acero quirúrgico, cuchillería, maxtractor, cafetera, expertea, easy release, kit de venta, usos y cuidados, agenda efectiva, productividad

**Manual de Ventas y/o Manual del Distribuidor:**
Palabras clave: requisitos comerciales, reactivación, contrato distribución, restricciones, ética, normas, áreas consignadas, áreas cerradas, distribuidor fallecido, oficina, reglas, ferias, descalificaciones, rescisión

**REGLAS IMPORTANTES:**
- Si hay múltiples coincidencias, combina fuentes y cita cada segmento por separado
- NUNCA generes respuestas especulativas ni uses información externa
- Siempre mantén el foco en la formación y el desarrollo profesional del distribuidor
- Proporciona información práctica y aplicable`
        },
        ...history.slice(-4), // Limitamos el historial para evitar tokens excesivos
        {
          role: "user",
          content: `Pregunta: "${message}"\n\nContexto:\n${context}`
        }
      ],
      temperature: 0.7,
      max_tokens: 500,
    });

    let answer = completion.choices[0].message.content;
    console.log("✅ Respuesta generada correctamente");

    // Preparar el enlace a WhatsApp
    const whatsappNumber = "2657218215";
    const whatsappMessage = encodeURIComponent("Hola, vengo del chatbot de manuales y necesito ayuda adicional.");
    const whatsappLink = `https://wa.me/${whatsappNumber}?text=${whatsappMessage}`;

    // Agregar enlace al manual si no lo tiene ya
    if (topManual && !answer.includes("Ver manual:")) {
      const manualLink = topManual.info.isRemote
        ? topManual.info.path  // Usar la URL directa para manuales remotos
        : `/manuals/${path.basename(topManual.info.path)}`; // Para manuales locales

      answer += `\n\n---\n[Ver manual: ${topManual.info.title}](${manualLink})`;
    }

    // Agregar enlace a WhatsApp para contacto con un asesor real
    if (!answer.includes("WhatsApp")) {
      answer += `\n\n¿Necesitas hablar con un asesor real? [Contacta por WhatsApp](${whatsappLink})`;
    }

    res.json({
      answer,
      manualInfo: topManual,
      contextUsado: contextResults.length,
      manualesCargados
    });
  } catch (err) {
    console.error("❌ Error en proceso de chat:", err);
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
    version: "1.2.0"
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

// Función para iniciar el servidor con manejo de errores de puerto
const startServer = (port) => {
  app.listen(port, () => {
    console.log(`🚀 Servidor ejecutándose en http://localhost:${port}`);
  }).on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
      console.log(`Puerto ${port} en uso, intentando con ${port + 1}...`);
      startServer(port + 1);
    } else {
      console.error('Error al iniciar el servidor:', error);
    }
  });
};

// Precargar los manuales antes de iniciar el servidor
precargarManuales().then(success => {
  startServer(PORT);
  if (success) {
    console.log(`📘 ${manualesInfo.length} manuales precargados y listos para consultas (${totalChunks} fragmentos totales)`);
  } else {
    console.log("⚠️ No se precargaron manuales correctamente. Verifica la configuración de manuales remotos o el directorio local.");
  }
});