require('dotenv').config();
const express = require('express');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const mongoose = require('mongoose');
const axios = require('axios');
const crypto = require('crypto');
const cors = require('cors');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 3000;
const NGC_API_KEY = process.env.NGC_API_KEY;
const MONGO_URI = process.env.MONGO_URI;
const RAG_SERVICE = process.env.RAG_SERVICE_URL || 'http://localhost:5001';

if (!NGC_API_KEY || !MONGO_URI) {
  console.error('Missing environment variables: NGC_API_KEY or MONGO_URI');
  process.exit(1);
}

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const uploadLimiter = rateLimit({ windowMs: 2 * 60 * 1000, max: 1,
  message: 'Please wait 2 minutes before uploading another document.' });
app.use('/upload', uploadLimiter);

const upload = multer({ storage: multer.memoryStorage() });

mongoose.connect(MONGO_URI)
  .then(() => console.log('Connected to MongoDB Atlas'))
  .catch(err => console.error('MongoDB error:', err.message));

const docSchema = new mongoose.Schema({
  docId: String, title: String, text: String, summary: String,
  expiry: { type: Date, expires: '24h' }
});
const Document = mongoose.model('Document', docSchema);

// Safety: sanitized logger - never logs document text
function safeLog(label, obj) {
  const s = { ...obj };
  if (s.text) s.text = `[REDACTED ${s.text.length} chars]`;
  console.log(label, JSON.stringify(s));
}
 
// Core LLM caller
async function callLlama(messages, { temperature = 0.3, top_p = 0.9, max_tokens = 600 } = {}) {
  const resp = await axios.post(
    'https://integrate.api.nvidia.com/v1/chat/completions',
    { model: 'meta/llama3-70b-instruct', messages, max_tokens, temperature, top_p },
    { headers: { Authorization: `Bearer ${NGC_API_KEY}`, 'Content-Type': 'application/json' }, timeout: 60000 }
  );
  return resp.data.choices?.[0]?.message?.content?.trim();
}

// Summarizer
async function summarizeDocument(text) {
  const prompt = `Summarize the provided legal document in under 300 words using bullet points under:
- Parties Involved
- Purpose and Scope
- Key Obligations and Rights
- Financial Terms (if applicable)
- Duration and Termination (if applicable)
- Legal and Compliance Requirements
- Dispute Resolution and Jurisdiction (if applicable)
- Other Critical Clauses
Include exact figures and document-specific terms. Be concise, professional, neutral.`;
  return callLlama([{ role: 'user', content: `${prompt}\n\nText:\n\n${text}` }], { temperature: 0.3, max_tokens: 400 });
}

// RAG Q&A
async function ragAnswer(docId, question, docText) {
  // Index document into vector store
  await axios.post(`${RAG_SERVICE}/index`, { doc_id: docId, text: docText });
  // Retrieve relevant chunks
  const { data } = await axios.post(`${RAG_SERVICE}/retrieve`, { doc_id: docId, question });
  const context = data.chunks.join('\n\n---\n\n');
  const prompt = `You are a legal assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I cannot find that information in the document." Do not fabricate.

Context:
${context}

Question: ${question}`;
  const answer = await callLlama([{ role: 'user', content: prompt }], { temperature: 0.1, max_tokens: 400 });
  return { answer, chunks_used: data.chunks.length };
}

// Agent: tool definitions
const LEGAL_TERMS = {
  indemnity: 'An obligation to compensate for loss or damage suffered by another party.',
  arbitration: 'A private dispute resolution process outside of court with a binding decision.',
  'force majeure': 'A clause freeing parties from obligations due to extraordinary, unforeseeable events.',
  jurisdiction: 'The legal authority of a court to hear a case, defined by geography or subject matter.',
  liability: 'Legal responsibility for actions or omissions that cause harm or loss.',
  'breach of contract': 'Failure by one party to fulfill their obligations under a contract.',
  'liquidated damages': 'A pre-agreed sum payable upon a specific breach of contract.',
  novation: 'The replacement of one party or obligation in a contract with a new one.',
  waiver: 'The voluntary relinquishment of a known right.',
  confidentiality: 'An obligation to keep information private and not share it with third parties.',
  termination: 'The ending of a contract by completion, mutual agreement, or breach.',
  'governing law': 'The law of a specific jurisdiction used to interpret the contract.',
};

function lookupLegalTerm(term) {
  return LEGAL_TERMS[term.toLowerCase().trim()] || `Term "${term}" not found in the legal glossary.`;
}

// Agent: ReAct-style loop
async function runAgent(docId, userRequest, doc) {
  const toolDescriptions = `You have access to these tools:
1. summarize_document() — Returns the document summary
2. lookup_legal_term(term) — Returns the definition of a legal term
3. answer_question(question) — Searches the document via RAG and answers a specific question

Respond ONLY with JSON:
{ "thought": "why you chose this tool", "tool": "tool_name", "input": "input or null" }`;

  const planRaw = await callLlama([
    { role: 'system', content: toolDescriptions },
    { role: 'user', content: `User request: "${userRequest}"\n\nWhich tool should I use?` }
  ], { temperature: 0.1, max_tokens: 200 });

  let plan;
  try { plan = JSON.parse(planRaw.replace(/```json|```/g, '').trim()); }
  catch { plan = { tool: 'answer_question', input: userRequest, thought: 'Fallback to Q&A' }; }

  console.log('Agent chose tool:', plan.tool);

  let toolResult;
  if (plan.tool === 'summarize_document') {
    toolResult = doc.summary || 'No summary available yet.';
  } else if (plan.tool === 'lookup_legal_term') {
    toolResult = lookupLegalTerm(plan.input || '');
  } else {
    const r = await ragAnswer(docId, plan.input || userRequest, doc.text);
    toolResult = r.answer;
  }

  const finalAnswer = await callLlama([
    { role: 'user', content: `User asked: "${userRequest}"\nTool: ${plan.tool}\nResult: ${toolResult}\n\nWrite a clear, helpful final answer.` }
  ], { temperature: 0.2, max_tokens: 400 });

  return { thought: plan.thought, tool_used: plan.tool, tool_input: plan.input, tool_result: toolResult, final_answer: finalAnswer };
}

// ── Routes ──────────────────────────────────────────────────────────────────

app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'Please upload a valid PDF file' });
    const { text: pdfText } = await pdfParse(req.file.buffer);
    if (pdfText.trim().length < 50) return res.status(400).json({ error: 'PDF text too short' });

    const docId = crypto.createHash('sha256').update(pdfText).digest('hex');
    const title = req.file.originalname;

    let doc = await Document.findOne({ docId });
    if (doc?.summary?.trim().length > 0 && doc.summary !== 'No summary generated') {
      axios.post(`${RAG_SERVICE}/index`, { doc_id: docId, text: doc.text }).catch(() => {});
      return res.json({ docId, title, summary: doc.summary });
    }

    if (!doc) doc = new Document({ docId, title, text: pdfText });
    else { doc.title = title; doc.text = pdfText; }
    await doc.save();

    const [summary] = await Promise.all([
      summarizeDocument(pdfText),
      axios.post(`${RAG_SERVICE}/index`, { doc_id: docId, text: pdfText }).catch(e =>
        console.warn('RAG service offline, skipping index:', e.message))
    ]);

    doc.summary = `# Summary of the Legal Document\n\n${summary}`;
    await doc.save();
    safeLog('Summarized:', { docId, title });
    return res.json({ docId, title, summary: doc.summary });
  } catch (err) {
    console.error('/upload error:', err.message);
    return res.status(500).json({ error: 'Server error. Please try again.' });
  }
});

app.post('/qa', async (req, res) => {
  try {
    const { docId, question } = req.body;
    if (!docId || !question) return res.status(400).json({ error: 'docId and question required' });
    const doc = await Document.findOne({ docId });
    if (!doc) return res.status(404).json({ error: 'Document not found. Upload it first.' });
    const result = await ragAnswer(docId, question, doc.text);
    return res.json(result);
  } catch (err) {
    if (err.response?.data?.error) return res.status(400).json({ error: err.response.data.error });
    console.error('/qa error:', err.message);
    return res.status(500).json({ error: 'Q&A failed. Please try again.' });
  }
});

app.post('/agent', async (req, res) => {
  try {
    const { docId, request: userRequest } = req.body;
    if (!docId || !userRequest) return res.status(400).json({ error: 'docId and request required' });
    const doc = await Document.findOne({ docId });
    if (!doc) return res.status(404).json({ error: 'Document not found.' });
    const result = await runAgent(docId, userRequest, doc);
    return res.json(result);
  } catch (err) {
    console.error('/agent error:', err.message);
    return res.status(500).json({ error: 'Agent failed. Please try again.' });
  }
});

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));

// eval-upload: accepts raw text (used by eval suite - no PDF needed)
app.post('/eval-upload', async (req, res) => {
  try {
    const { text, title } = req.body;
    if (!text || text.trim().length < 20) return res.status(400).json({ error: 'Text too short' });
    const docId = crypto.createHash('sha256').update(text).digest('hex');
    let doc = await Document.findOne({ docId });
    if (doc?.summary) {
      return res.json({ docId, title, summary: doc.summary });
    }
    if (!doc) doc = new Document({ docId, title: title || 'eval_doc', text });
    else { doc.text = text; }
    const [summary] = await Promise.all([
      summarizeDocument(text),
      axios.post(`${RAG_SERVICE}/index`, { doc_id: docId, text }).catch(() => {})
    ]);
    doc.summary = `# Summary\n\n${summary}`;
    await doc.save();
    return res.json({ docId, title, summary: doc.summary });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});
