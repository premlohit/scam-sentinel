// Tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.querySelector(`.tab-content[data-content="${tab.dataset.tab}"]`).classList.add('active');
  });
});

// PDF dropzone
const dropzone = document.getElementById('dropzone');
const pdfInput = document.getElementById('pdfFile');
const dzFilename = document.getElementById('dzFilename');

dropzone.addEventListener('click', () => pdfInput.click());
pdfInput.addEventListener('change', () => {
  if (pdfInput.files.length) dzFilename.textContent = pdfInput.files[0].name;
});
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = 'var(--accent)'; });
dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; });
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files.length) {
    pdfInput.files = e.dataTransfer.files;
    dzFilename.textContent = e.dataTransfer.files[0].name;
  }
});

const scanBtn = document.getElementById('scanBtn');
const errorMsg = document.getElementById('errorMsg');
const resultPanel = document.getElementById('resultPanel');

scanBtn.addEventListener('click', async () => {
  errorMsg.textContent = '';
  const activeTab = document.querySelector('.tab.active').dataset.tab;
  const formData = new FormData();

  if (activeTab === 'upload') {
    if (!pdfInput.files.length) {
      errorMsg.textContent = 'Please choose a PDF file first.';
      return;
    }
    formData.append('pdf', pdfInput.files[0]);
  } else {
    const text = document.getElementById('jobText').value.trim();
    if (!text) {
      errorMsg.textContent = 'Please paste a job description first.';
      return;
    }
    formData.append('job_text', text);
  }

  formData.append('company', document.getElementById('company').value.trim());
  formData.append('email', document.getElementById('email').value.trim());
  formData.append('website', document.getElementById('website').value.trim());

  scanBtn.disabled = true;
  scanBtn.querySelector('.scan-btn-label').textContent = 'Scanning...';

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      errorMsg.textContent = data.error || 'Something went wrong.';
      return;
    }
    renderResult(data);
  } catch (err) {
    errorMsg.textContent = 'Network error — is the server running?';
  } finally {
    scanBtn.disabled = false;
    scanBtn.querySelector('.scan-btn-label').textContent = 'Scan Posting';
  }
});

function renderResult(data) {
  resultPanel.hidden = false;
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const risk = data.risk_score;
  document.getElementById('riskNumber').textContent = risk;
  document.getElementById('mlConfidence').textContent = data.ml_confidence + '% probability fake (model)';

  const verdict = document.getElementById('verdictLabel');
  verdict.textContent = data.prediction === 'Fake' ? '⚠ Likely FAKE' : '✓ Likely REAL';
  verdict.className = 'verdict ' + (data.prediction === 'Fake' ? 'fake' : 'real');

  document.getElementById('explanationText').textContent = data.explanation;

  // Gauge
  const circumference = 283;
  const offset = circumference - (risk / 100) * circumference;
  const fill = document.getElementById('gaugeFill');
  fill.style.strokeDashoffset = offset;
  fill.style.stroke = risk >= 65 ? 'var(--danger)' : risk >= 35 ? 'var(--warn)' : 'var(--safe)';

  fillList('redFlagsList', data.red_flags);
  fillList('salaryFlagsList', data.salary_flags);
  fillList('verificationFlagsList', data.verification_flags);

  const wordsBlock = document.getElementById('wordsBlock');
  const topWords = document.getElementById('topWords');
  topWords.innerHTML = '';
  if (data.top_contributing_words && data.top_contributing_words.length) {
    wordsBlock.hidden = false;
    data.top_contributing_words.forEach(w => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = w;
      topWords.appendChild(chip);
    });
  } else {
    wordsBlock.hidden = true;
  }
}

function fillList(id, items) {
  const el = document.getElementById(id);
  el.innerHTML = '';
  if (!items || !items.length) {
    const li = document.createElement('li');
    li.className = 'none';
    li.textContent = 'No issues detected';
    el.appendChild(li);
    return;
  }
  items.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    el.appendChild(li);
  });
}
