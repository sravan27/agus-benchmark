document.addEventListener('DOMContentLoaded', async () => {
  setupScrollStaggering();

  try {
    const response = await fetch('data.json');
    if (!response.ok) throw new Error('Data load failed');
    const data = await response.json();
    
    populateHeroMetrics(data.models);
    populateTable(data.models);
    populateInsights(data.insights);
    populateFailureModes(data.failureModes);
    
    // Retrigger observer evaluation for dynamically injected elements
    setTimeout(() => {
      document.querySelectorAll('.stagger-card, .insight-card, .diagnostic-strip').forEach(el => {
        el.classList.add('stagger-item');
      });
    }, 100);

  } catch (error) {
    console.error('Error loading data:', error);
  }
});

function formatNumber(num) {
  if (num === null || num === undefined) return '—';
  return num.toFixed(4);
}

function setupScrollStaggering() {
  const sections = document.querySelectorAll('.stagger-section');
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        // Trigger children
        entry.target.querySelectorAll('.stagger-item, .stagger-card').forEach(child => {
          child.classList.add('in-view');
        });
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.15,
    rootMargin: "0px 0px -100px 0px"
  });

  sections.forEach(sec => observer.observe(sec));
}

function populateHeroMetrics(models) {
  const llama = models.find(m => m.name.includes("Llama"));
  const qwen = models.find(m => m.name.includes("Qwen"));
  const mistral = models.find(m => m.name.includes("Mistral"));

  if(llama) document.getElementById('metric-llama-static').innerText = formatNumber(llama.static_accuracy);
  if(qwen) document.getElementById('metric-qwen-traj').innerText = formatNumber(qwen.belief_trajectory_quality);
  if(mistral) document.getElementById('metric-mistral-cf').innerText = formatNumber(mistral.counterfactual_update_fidelity);
}

function populateTable(models) {
  const tbody = document.querySelector('#results-table tbody');
  if (!tbody) return;
  
  models.forEach(model => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="model-name">${model.name}</td>
      <td>${formatNumber(model.static_accuracy)}</td>
      <td>${formatNumber(model.belief_trajectory_quality)}</td>
      <td>${formatNumber(model.trajectory_instability_index)}</td>
      <td>${formatNumber(model.counterfactual_update_fidelity)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateInsights(insights) {
  const container = document.getElementById('insights-grid');
  if (!container) return;
  
  insights.forEach(insight => {
    const card = document.createElement('div');
    card.className = 'insight-card stagger-item';
    card.innerHTML = `
      <h3>${insight.title}</h3>
      <p>${insight.description}</p>
    `;
    container.appendChild(card);
  });
}

function populateFailureModes(failureModes) {
  const container = document.getElementById('failure-modes-grid');
  if (!container) return;
  
  failureModes.forEach(mode => {
    const strip = document.createElement('div');
    strip.className = 'diagnostic-strip stagger-item';
    strip.innerHTML = `
      <div class="diagnostic-header">
        <h3>${mode.label}</h3>
        <div class="diagnostic-implication">${mode.implication}</div>
      </div>
      <div class="diagnostic-body">
        <p>${mode.description}</p>
      </div>
    `;
    container.appendChild(strip);
  });
}
