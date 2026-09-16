const EMPTY_MESSAGE = 'No alerts available yet.';

const escapeHtml = (value) => {
  if (value === null || value === undefined) {
    return '';
  }

  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
};

const safeNumber = (value, fallback = 0) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
};

const setText = (elementId, value) => {
  const element = document.getElementById(elementId);
  if (!element) {
    return;
  }
  element.textContent = value ?? 'N/A';
};

const updateSummary = (summary = {}) => {
  setText('trains-requiring-attention', safeNumber(summary.trains_requiring_attention, 0));
  setText('critical-anomalies', safeNumber(summary.critical_anomalies, 0));
  setText('developing-warnings', safeNumber(summary.developing_warnings, 0));
  setText('highest-priority-component', summary.highest_priority_component ?? 'N/A');
};

const updateStatus = (metadata = {}) => {
  const statusElement = document.getElementById('data-status');
  const status = metadata.data_status || 'Placeholder';
  if (statusElement) {
    statusElement.textContent = `Data status: ${status}`;
  }
};

const severityClassMap = {
  critical: 'severity-critical',
  warning: 'severity-warning',
  minor: 'severity-minor',
  normal: 'severity-normal'
};

const renderAlertList = (alerts = []) => {
  const alertList = document.getElementById('alert-list');
  if (!alertList) {
    return;
  }

  if (!alerts.length) {
    alertList.innerHTML = `<p class="empty-state">${EMPTY_MESSAGE}</p>`;
    return;
  }

  alertList.innerHTML = alerts
    .map((alert, index) => {
      const title = escapeHtml(alert.title || alert.component || `Alert ${index + 1}`);
      const severity = (alert.severity || 'normal').toLowerCase();
      const component = escapeHtml(alert.component || 'Unknown component');
      const detail = escapeHtml(alert.detail || 'No supporting evidence yet.');
      const className = severityClassMap[severity] || severityClassMap.normal;

      return `
        <button class="alert-row" data-index="${index}" type="button">
          <div class="alert-header">
            <span class="alert-title">${title}</span>
            <span class="severity-badge ${className}">${escapeHtml(severity)}</span>
          </div>
          <div class="alert-meta">${component}</div>
          <div class="alert-meta">${detail}</div>
        </button>
      `;
    })
    .join('');

  const rows = alertList.querySelectorAll('.alert-row');
  rows.forEach((row) => {
    row.addEventListener('click', () => {
      rows.forEach((item) => item.classList.remove('active'));
      row.classList.add('active');
      const selectedIndex = Number(row.dataset.index);
      const selectedAlert = alerts[selectedIndex];
      renderExplanation(selectedAlert);
      updateChart(selectedAlert?.timeseries || []);
    });
  });

  if (alerts[0]) {
    renderExplanation(alerts[0]);
    updateChart(alerts[0].timeseries || []);
  }
};

const renderExplanation = (alert = null) => {
  const explanation = document.getElementById('explanation-content');
  if (!explanation) {
    return;
  }

  if (!alert) {
    explanation.innerHTML = '<p class="empty-state">Select an alert to view the supporting evidence.</p>';
    return;
  }

  const summary = alert.summary || 'No supporting detail available yet.';
  const evidence = alert.evidence || [];
  const evidenceHtml = evidence.length
    ? evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join('')
    : '<li>No supporting evidence recorded yet.</li>';

  explanation.innerHTML = `
    <div class="explanation-content">
      <p><strong>Component:</strong> ${escapeHtml(alert.component || 'Unknown')}</p>
      <p><strong>Severity:</strong> ${escapeHtml((alert.severity || 'normal').toUpperCase())}</p>
      <p><strong>Summary:</strong> ${escapeHtml(summary)}</p>
      <ul>${evidenceHtml}</ul>
    </div>
  `;
};

const updateChart = (series = []) => {
  const chartElement = document.getElementById('temperature-chart');
  const emptyState = document.getElementById('chart-empty-state');

  if (!chartElement) {
    return;
  }

  const chartContext = chartElement.getContext('2d');

  if (!series.length) {
    if (emptyState) {
      emptyState.style.display = 'block';
    }
    if (window.nebulaChartInstance) {
      window.nebulaChartInstance.destroy();
      window.nebulaChartInstance = null;
    }
    return;
  }

  if (emptyState) {
    emptyState.style.display = 'none';
  }

  const labels = series.map((point) => point.label ?? point.timestamp ?? '');
  const values = series.map((point) => safeNumber(point.value, 0));

  if (window.nebulaChartInstance) {
    window.nebulaChartInstance.destroy();
  }

  window.nebulaChartInstance = new Chart(chartContext, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Temperature',
        data: values,
        borderColor: '#5ec9ff',
        backgroundColor: 'rgba(94, 201, 255, 0.18)',
        tension: 0.25,
        fill: false,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: {
            color: '#9eb7c8'
          },
          grid: {
            color: 'rgba(158, 183, 200, 0.15)'
          }
        },
        y: {
          ticks: {
            color: '#9eb7c8'
          },
          grid: {
            color: 'rgba(158, 183, 200, 0.15)'
          }
        }
      }
    }
  });
};

const loadDashboard = async () => {
  try {
    const response = await fetch('./data/dashboard_data.json');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    updateStatus(payload.metadata || {});
    updateSummary(payload.summary || {});
    renderAlertList(payload.alerts || []);
    if (payload.timeseries && payload.timeseries.length) {
      updateChart(payload.timeseries[0] || []);
    } else {
      updateChart([]);
    }
  } catch (error) {
    console.error('Dashboard data could not be loaded:', error);
    updateStatus({ data_status: 'Error loading data' });
    updateSummary({
      trains_requiring_attention: 0,
      critical_anomalies: 0,
      developing_warnings: 0,
      highest_priority_component: null
    });
    renderAlertList([]);
    updateChart([]);
  }
};

document.addEventListener('DOMContentLoaded', loadDashboard);
