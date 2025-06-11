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

if (!NGC_API_KEY || !MONGO_URI) {
  console.error('❌ Missing environment variables: NGC_API_KEY or MONGO_URI');
  process.exit(1);
}

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Rate limiting: 1 request per 2 minutes per IP
const uploadLimiter = rateLimit({
  windowMs: 2 * 60 * 1000, // 2 minutes
  max: 1, // 1 request
  message: 'Please wait 2 minutes before uploading another document.'
});
app.use('/upload', uploadLimiter);

// Multer setup
const upload = multer({ storage: multer.memoryStorage() });

// MongoDB connection
mongoose.connect(MONGO_URI)
  .then(() => console.log('✅ Connected to MongoDB Atlas'))
  .catch(err => console.error('❌ MongoDB Atlas connection error:', err.message));

// Document schema
const docSchema = new mongoose.Schema({
  docId: String,
  title: String,
  text: String,
  summary: String,
  expiry: { type: Date, expires: '24h' }
});
const Document = mongoose.model('Document', docSchema);

// Summarize with Llama-3.1-70B-Instruct
async function summarizeWithRetry(text, retries = 2) {
  const prompt = `Summarize the provided legal document in under 300 words using bullet points under the following headings, adapting to the document’s content and including specific figures or terms when present:
- Parties Involved: Identify the main individuals or entities signing or referenced (e.g., names, roles).
- Purpose and Scope: Describe the document’s primary objective or subject matter (e.g., service, lease, affidavit).
- Key Obligations and Rights: List major responsibilities, duties, or entitlements of the parties (e.g., deliverables, indemnification).
- Financial Terms (if applicable): Detail specific payments, fees, deposits, or financial considerations (e.g., amounts, schedules).
- Duration and Termination (if applicable): Specify the term, validity, or conditions for ending the agreement (e.g., notice periods).
- Legal and Compliance Requirements: Note specific laws, regulations, or compliance obligations (e.g., notarization, licensing).
- Dispute Resolution and Jurisdiction (if applicable): Outline how disputes are handled and governing law (e.g., arbitration, state).
- Other Critical Clauses: Include unique or significant terms (e.g., subcontracting, notarization, record retention).

Focus on essential details, including exact figures (e.g., $ amounts, days) and document-specific terms (e.g., exhibits). Avoid boilerplate or repetitive language. Ensure the summary is concise, professional, and neutral, reflecting the document’s type (e.g., rent agreement, affidavit) if identifiable.`;

  const url = 'https://integrate.api.nvidia.com/v1/chat/completions';
  const headers = {
    Authorization: `Bearer ${NGC_API_KEY}`,
    'Content-Type': 'application/json'
  };
  const payload = {
    model: 'meta/llama3-70b-instruct',
    messages: [{
      role: 'user',
      content: `${prompt}\n\nText:\n\n${text}`
    }],
    max_tokens: 400, // ~300 words
    temperature: 0.3,
    top_p: 0.9
  };

  for (let i = 0; i < retries; i++) {
    try {
      console.log(`Sending ${text.length} characters to NVIDIA API`);
      const resp = await axios.post(url, payload, { headers, timeout: 60000 });
      if (resp.data.choices?.[0]?.message?.content) {
        const summary = resp.data.choices[0].message.content.trim();
        console.log(`✅ Generated summary: ${summary.split(/\s+/).length} words`);
        return summary;
      }
      throw new Error('Unexpected response format');
    } catch (err) {
      if (i < retries - 1) {
        console.warn(`Retrying summarization (attempt ${i + 1})`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      } else {
        console.error('🔴 NVIDIA API error:', err.response?.data || err.message);
        throw err;
      }
    }
  }
}

// Word count helper
function wordCount(text) {
  return text.split(/\s+/).filter(word => word.length > 0).length;
}

// Upload and summarize
app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    console.log('📄 Received /upload request');
    if (!req.file) return res.status(400).json({ error: 'Please upload a valid PDF file' });

    const { text: pdfText } = await pdfParse(req.file.buffer);
    console.log(`✂️ Extracted ${pdfText.length} characters from PDF`);

    if (pdfText.trim().length < 50) return res.status(400).json({ error: 'PDF text too short' });

    const docId = crypto.createHash('sha256').update(pdfText).digest('hex');
    const title = req.file.originalname;

    let existing = await Document.findOne({ docId });
    if (existing && existing.summary && existing.summary.trim().length > 0 && existing.summary !== '⚠️ No summary generated') {
      console.log('📜 Returning cached summary');
      return res.json({ docId, title, summary: existing.summary });
    }

    if (!existing) {
      existing = new Document({ docId, title, text: pdfText });
    } else {
      existing.title = title;
      existing.text = pdfText;
    }
    await existing.save();
    console.log(`💾 Cached document: ${docId}`);

    // Summarize the entire document
    let summary = await summarizeWithRetry(pdfText);

    // Ensure summary is under 300 words
    if (wordCount(summary) > 300) {
      const words = summary.split(/\s+/).filter(word => word.length > 0);
      summary = words.slice(0, 300).join(' ') + '...';
      console.log(`✂️ Truncated summary to 300 words`);
    }

    // Format summary with heading
    summary = `# Summary of the Legal Document\n\n${summary}`;

    existing.summary = summary || '⚠️ No summary generated';
    await existing.save();

    console.log('📤 Returning summary');
    return res.json({ docId, title, summary: existing.summary });
  } catch (error) {
    console.error('❌ /upload error:', error.message);
    return res.status(500).json({ error: 'Server error. Please try again later.' });
  }
});

// Start server
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));