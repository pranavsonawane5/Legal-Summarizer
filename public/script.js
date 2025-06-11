// public/script.js
const terms = {
  indemnity: 'Compensation for loss or damage',
  clause: 'A section of a contract',
  breach: 'Violation of contract terms'
};

async function fetchRecentDocs() {
  try {
    const response = await fetch('/recent');
    const docs = await response.json();
    const select = document.getElementById('recentDocs');
    select.innerHTML = '<option value="">Select Recent Document</option>';
    docs.forEach(doc => {
      const option = document.createElement('option');
      option.value = doc.docId;
      option.text = doc.title;
      select.appendChild(option);
    });
  } catch (error) {
    console.error('Error fetching recent docs:', error);
  }
}

async function summarize() {
  const fileInput = document.getElementById('pdfInput');
  const selectedDocId = document.getElementById('recentDocs').value;

  if (!fileInput.files[0] && !selectedDocId) {
    return alert('Please upload a PDF or select a recent document');
  }

  try {
    let response, data;
    if (fileInput.files[0]) {
      const formData = new FormData();
      formData.append('pdf', fileInput.files[0]);
      response = await fetch('/upload', { method: 'POST', body: formData });
    } else {
      response = await fetch('/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ docId: selectedDocId })
      });
    }

    data = await response.json();
    if (data.error) {
      console.error('Server error:', data.error);
      return alert(`Error: ${data.error}`);
    }

    localStorage.setItem('docId', data.docId);
    document.getElementById('summary').innerHTML =
      `<h2>Summary</h2><p>${highlightTerms(data.summary)}</p>`;
    fetchRecentDocs();
  } catch (error) {
    console.error('Error summarizing document:', error.message);
    alert(`Error summarizing document: ${error.message}`);
  }
}

async function askQuestion() {
  const questionText = document.getElementById('question').value.trim();
  const docId = localStorage.getItem('docId') || document.getElementById('recentDocs').value;

  if (!questionText || !docId) {
    console.error('Missing question or document');
    return alert('Please select a document and enter a question');
  }

  try {
    const response = await fetch('/qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ docId, question: questionText })
    });

    const data = await response.json();
    if (data.error) {
      console.error('Q&A error:', data.error);
      return alert(`Error: ${data.error}`);
    }

    document.getElementById('answer').innerHTML =
      `<h2>Answer</h2><p>${data.answer}</p>`;
  } catch (error) {
    console.error('Error answering question:', error.message);
    alert(`Error answering question: ${error.message}`);
  }
}

function highlightTerms(text) {
  Object.keys(terms).forEach(term => {
    text = text.replace(
      new RegExp(`\\b${term}\\b`, 'gi'),
      `<span class="term" title="${terms[term]}">${term}</span>`
    );
  });
  return text;
}

// Load recent documents on page load
fetchRecentDocs();
