/**
 * EEG Visualization & Preprocessing — Frontend Application
 * =========================================================
 * Interactive EEG visualization using Plotly.js
 * Light theme with scroll-zoom and dynamic range updates
 */

// ── Constants ────────────────────────────────────────────────
const CHANNEL_COLORS = [
    '#4f46e5', '#7c3aed', '#8b5cf6', '#a855f7',
    '#d946ef', '#ec4899', '#f43f5e', '#ea580c',
    '#d97706', '#059669', '#0d9488', '#0891b2',
    '#0284c7', '#2563eb', '#4f46e5', '#6366f1',
];

const PLOTLY_LAYOUT_DEFAULTS = {
    paper_bgcolor: 'rgba(255,255,255,0)',
    plot_bgcolor: 'rgba(248, 250, 252, 0.8)',
    font: { family: 'Inter, sans-serif', color: '#475569', size: 11 },
    margin: { l: 60, r: 20, t: 30, b: 40 },
    xaxis: {
        gridcolor: 'rgba(99,102,241,0.08)',
        zerolinecolor: 'rgba(99,102,241,0.15)',
        linecolor: 'rgba(0,0,0,0.1)',
    },
    yaxis: {
        gridcolor: 'rgba(99,102,241,0.08)',
        zerolinecolor: 'rgba(99,102,241,0.15)',
        linecolor: 'rgba(0,0,0,0.1)',
    },
    legend: {
        bgcolor: 'rgba(255,255,255,0.9)',
        bordercolor: 'rgba(99,102,241,0.15)',
        borderwidth: 1,
        font: { size: 10, color: '#475569' },
    },
};

// ── State ────────────────────────────────────────────────────
let appState = {
    loaded: false,
    channelNames: [],
    visibleChannels: {},
    totalDuration: 0,
    sfreq: 125,
    viewStart: 0,
    viewWindow: 10, // seconds to display at once
    isPlotting: false,
    liveRunning: false,
    livePollTimer: null,
    replayRunning: false,
    replayTimer: null,
    replayLastTs: 0,
    replaySpeed: 1.0,
    loadedFilePath: null,
    workflowStage: 'ready', // ready | loaded | processed | analyzed
};

// ── API Helpers ──────────────────────────────────────────────
async function api(endpoint, method = 'GET', body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(endpoint, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'API error');
    return data;
}

function showLoading(text = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    overlay.querySelector('.loading-text').textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

// ── Initialize ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    initDarkMode();
    await loadFileList();
    await refreshLiveStatus();
    setupTabs();
    setupEventListeners();
    setupKeyboardShortcuts();
    updateWorkflowBreadcrumb('ready');
});

function stopReplay() {
    appState.replayRunning = false;
    if (appState.replayTimer) {
        cancelAnimationFrame(appState.replayTimer);
        appState.replayTimer = null;
    }
    const btn = document.getElementById('btn-replay-toggle');
    if (btn) btn.textContent = '▶ Play';
}

function replayTick(nowMs) {
    if (!appState.replayRunning || !appState.loaded || appState.liveRunning) return;

    if (appState.replayLastTs <= 0) {
        appState.replayLastTs = nowMs;
    }
    const dt = Math.max(0, (nowMs - appState.replayLastTs) / 1000.0);
    appState.replayLastTs = nowMs;

    const maxStart = Math.max(0, appState.totalDuration - appState.viewWindow);
    appState.viewStart = Math.min(maxStart, appState.viewStart + dt * appState.replaySpeed);
    document.getElementById('view-start').value = appState.viewStart.toFixed(2);

    refreshPlot();
    updateProgressBar();
    if (appState.viewStart >= maxStart) {
        stopReplay();
        return;
    }
    appState.replayTimer = requestAnimationFrame(replayTick);
}

function toggleReplay() {
    if (!appState.loaded || appState.liveRunning) return;
    if (appState.replayRunning) {
        stopReplay();
        return;
    }
    appState.replayRunning = true;
    appState.replayLastTs = 0;
    const btn = document.getElementById('btn-replay-toggle');
    if (btn) btn.textContent = '⏸ Pause';
    appState.replayTimer = requestAnimationFrame(replayTick);
}

function setReplaySpeed() {
    const val = parseFloat(document.getElementById('replay-speed').value);
    if (!isNaN(val) && val > 0) {
        appState.replaySpeed = val;
    }
}

// ── File Loading ─────────────────────────────────────────────
async function loadFileList() {
    try {
        const data = await api('/api/files');
        const list = document.getElementById('file-list');
        list.innerHTML = '';

        if (data.files.length === 0) {
            list.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;padding:8px;">No data files found in raw_data/</div>';
            return;
        }

        data.files.forEach(f => {
            const item = document.createElement('div');
            const isActive = appState.loadedFilePath === f.path;
            item.className = 'file-item' + (isActive ? ' active' : '');
            const icon = f.name.toLowerCase().endsWith('.csv') ? '📊' : '📄';
            item.innerHTML = `
                <span class="file-icon">${icon}</span>
                <div class="file-info">
                    <div class="file-name" title="${f.name}">${f.name}</div>
                    <div class="file-size">${f.size_mb} MB</div>
                </div>
                <span class="file-check">✓</span>
            `;
            item.onclick = () => loadFile(f.path, item);
            list.appendChild(item);
        });
    } catch (e) {
        showToast('Failed to list files: ' + e.message, 'error');
    }
}

async function loadFile(filepath, itemElement) {
    stopReplay();
    showLoading('Loading EEG data...');
    try {
        // Highlight active file
        document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
        if (itemElement) itemElement.classList.add('active');

        const data = await api('/api/load', 'POST', { filepath });

        if (data.success) {
            appState.loaded = true;
            appState.channelNames = data.info.ch_names;
            appState.totalDuration = data.info.duration_sec;
            appState.sfreq = data.info.sfreq;
            appState.viewStart = 0;

            // Initialize all channels as visible
            appState.visibleChannels = {};
            data.info.ch_names.forEach(ch => appState.visibleChannels[ch] = true);
            appState.loadedFilePath = filepath;

            updateInfoPanel(data.info);
            buildChannelToggles(data.info.ch_names);
            updateHistoryPanel(['Loaded raw data']);
            enableControls(true);

            // Update status badge and workflow
            document.getElementById('status-text').textContent = 'Data Loaded';
            updateWorkflowBreadcrumb('loaded');

            // Load and plot
            await refreshPlot();
            updateProgressBar();
            showToast(`Loaded ${data.info.n_channels} channels, ${data.info.duration_sec}s`, 'success');
        }
    } catch (e) {
        showToast('Failed to load: ' + e.message, 'error');
    }
    hideLoading();
}

// ── Info Panel ───────────────────────────────────────────────
function updateInfoPanel(info) {
    document.getElementById('info-channels').textContent = info.n_channels;
    document.getElementById('info-samples').textContent = info.n_samples.toLocaleString();
    document.getElementById('info-sfreq').textContent = info.sfreq + ' Hz';
    document.getElementById('info-duration').textContent = info.duration_sec + 's';
    if (info.source_format === 'live_stream') {
        document.getElementById('info-format').textContent = 'Live Stream';
    } else {
        document.getElementById('info-format').textContent = info.source_format === 'openbci_txt' ? 'OpenBCI' : 'BrainFlow';
    }
}

// ── Channel Toggles ──────────────────────────────────────────
function buildChannelToggles(chNames) {
    const grid = document.getElementById('channel-grid');
    grid.innerHTML = '';

    chNames.forEach((ch, i) => {
        const label = document.createElement('label');
        label.className = 'channel-toggle';
        label.innerHTML = `
            <input type="checkbox" checked data-channel="${ch}">
            <span class="ch-color-dot" style="background:${CHANNEL_COLORS[i % 16]}"></span>
            <span>${ch}</span>
        `;
        label.querySelector('input').addEventListener('change', (e) => {
            appState.visibleChannels[ch] = e.target.checked;
            refreshPlot();
        });
        grid.appendChild(label);
    });
}

// ── Plotting ─────────────────────────────────────────────────
async function refreshPlot() {
    if (!appState.loaded || appState.isPlotting) return;
    appState.isPlotting = true;

    try {
        const tStart = appState.liveRunning ? Math.max(0, appState.totalDuration - appState.viewWindow) : appState.viewStart;
        const tEnd = appState.liveRunning ? -1 : (appState.viewStart + appState.viewWindow);
        const params = new URLSearchParams({
            t_start: tStart,
            t_end: tEnd,
            max_points: 8000,
        });

        const data = await api(`/api/channels?${params}`);
        const incomingChannels = Object.keys(data.channels || {});
        if (appState.liveRunning && appState.channelNames.length === 0 && incomingChannels.length > 0) {
            appState.channelNames = incomingChannels;
            appState.visibleChannels = {};
            incomingChannels.forEach(ch => appState.visibleChannels[ch] = true);
            buildChannelToggles(incomingChannels);
            updateInfoPanel({
                n_channels: incomingChannels.length,
                n_samples: 0,
                sfreq: data.sfreq,
                duration_sec: 0,
                source_format: 'live_stream',
            });
        }
        appState.totalDuration = data.total_duration;
        if (appState.liveRunning) {
            appState.viewStart = Math.max(0, appState.totalDuration - appState.viewWindow);
            document.getElementById('view-start').value = appState.viewStart.toFixed(1);
            const estSamples = Math.round(appState.totalDuration * data.sfreq);
            document.getElementById('info-samples').textContent = estSamples.toLocaleString();
            document.getElementById('info-duration').textContent = appState.totalDuration.toFixed(2) + 's';
        }
        plotTimeSeries(data);
    } catch (e) {
        if (!(appState.liveRunning && e.message && e.message.includes('No data loaded'))) {
            showToast('Plot error: ' + e.message, 'error');
        }
    }

    appState.isPlotting = false;
}

function plotTimeSeries(data) {
    const container = document.getElementById('timeseries-chart');
    const traces = [];
    const visibleNames = Object.keys(data.channels).filter(ch => appState.visibleChannels[ch]);
    const nVisible = visibleNames.length;

    if (nVisible === 0 || !data.time || data.time.length === 0) {
        Plotly.purge(container);
        return;
    }

    // Remove channel DC offsets for display and use robust spacing.
    const centeredByChannel = {};
    const channelScales = [];
    visibleNames.forEach(ch => {
        const raw = data.channels[ch] || [];
        if (raw.length === 0) return;
        const sorted = raw.slice().sort((a, b) => a - b);
        const med = sorted[Math.floor(sorted.length / 2)];
        const centered = raw.map(v => v - med);
        centeredByChannel[ch] = centered;

        const absVals = centered.map(v => Math.abs(v)).sort((a, b) => a - b);
        const p95 = absVals[Math.floor(0.95 * (absVals.length - 1))] || 0;
        channelScales.push(p95);
    });

    if (Object.keys(centeredByChannel).length === 0) {
        Plotly.purge(container);
        return;
    }

    const sortedScales = channelScales.slice().sort((a, b) => a - b);
    const medScale = sortedScales[Math.floor(sortedScales.length / 2)] || 50;
    const spacing = Math.max(100, medScale * 8);

    const useSpline = data.time.length <= 3000;

    visibleNames.forEach((ch, i) => {
        const chIdx = appState.channelNames.indexOf(ch);
        const offset = (nVisible - 1 - i) * spacing;
        const centered = centeredByChannel[ch] || [];
        const yData = centered.map(v => v + offset);

        traces.push({
            x: data.time,
            y: yData,
            type: useSpline ? 'scatter' : 'scattergl',
            mode: 'lines',
            name: ch,
            line: {
                color: CHANNEL_COLORS[chIdx % 16],
                width: 1.2,
                shape: useSpline ? 'spline' : 'linear',
                smoothing: useSpline ? 0.7 : 0,
            },
            hovertemplate: `<b>${ch}</b><br>Time: %{x:.3f}s<br>Value: %{customdata:.2f} µV<extra></extra>`,
            customdata: data.channels[ch],
        });
    });

    const layout = {
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: {
            text: `EEG Time Series — ${appState.viewStart.toFixed(1)}s to ${(appState.viewStart + appState.viewWindow).toFixed(1)}s`,
            font: { size: 13, color: '#475569' },
        },
        xaxis: {
            ...PLOTLY_LAYOUT_DEFAULTS.xaxis,
            title: { text: 'Time (s)', font: { size: 11, color: '#64748b' } },
            range: [data.time[0], data.time[data.time.length - 1]],
            fixedrange: false, // Allow horizontal zoom
        },
        yaxis: {
            ...PLOTLY_LAYOUT_DEFAULTS.yaxis,
            title: '',
            showticklabels: false,
            range: [-spacing, nVisible * spacing],
            fixedrange: false, // Allow vertical zoom
        },
        showlegend: true,
        legend: {
            ...PLOTLY_LAYOUT_DEFAULTS.legend,
            x: 1.01,
            y: 1,
        },
        height: Math.max(400, nVisible * 45 + 80),
        dragmode: 'zoom', // Default to zoom mode
    };

    // Add channel name annotations on y-axis
    const annotations = visibleNames.map((ch, i) => ({
        x: -0.01,
        y: (nVisible - 1 - i) * spacing,
        xref: 'paper',
        yref: 'y',
        text: ch,
        showarrow: false,
        font: { size: 10, color: CHANNEL_COLORS[appState.channelNames.indexOf(ch) % 16] },
        xanchor: 'right',
    }));
    layout.annotations = annotations;

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
        displaylogo: false,
        scrollZoom: true, // Enable scroll wheel zoom
        doubleClick: 'reset', // Double-click to reset zoom
    };

    Plotly.react(container, traces, layout, config);

    // Listen for zoom/pan events to sync view range
    container.removeAllListeners && container.removeAllListeners('plotly_relayout');
    container.on('plotly_relayout', (eventData) => {
        if (eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
            const newStart = Math.max(0, eventData['xaxis.range[0]']);
            const newEnd = Math.min(appState.totalDuration, eventData['xaxis.range[1]']);
            const newWindow = newEnd - newStart;

            // Only re-fetch if we zoomed significantly (more than 2x in or out)
            const ratio = newWindow / appState.viewWindow;
            if (ratio < 0.4 || ratio > 2.5) {
                appState.viewStart = newStart;
                appState.viewWindow = newWindow;
                document.getElementById('view-start').value = newStart.toFixed(1);
                document.getElementById('view-window').value = newWindow.toFixed(1);
                refreshPlot(); // Re-fetch data at new resolution
            }
        }
        // Handle reset (double-click)
        if (eventData['xaxis.autorange'] || eventData['yaxis.autorange']) {
            appState.viewStart = 0;
            appState.viewWindow = Math.min(10, appState.totalDuration);
            document.getElementById('view-start').value = '0';
            document.getElementById('view-window').value = appState.viewWindow.toFixed(1);
            refreshPlot();
        }
    });
}

// ── Navigation ───────────────────────────────────────────────
function navigate(direction) {
    if (!appState.loaded) return;
    stopReplay();

    const step = appState.viewWindow * 0.5;
    if (direction === 'prev') {
        appState.viewStart = Math.max(0, appState.viewStart - step);
    } else if (direction === 'next') {
        appState.viewStart = Math.min(
            appState.totalDuration - appState.viewWindow,
            appState.viewStart + step
        );
    } else if (direction === 'start') {
        appState.viewStart = 0;
    } else if (direction === 'end') {
        appState.viewStart = Math.max(0, appState.totalDuration - appState.viewWindow);
    }

    document.getElementById('view-start').value = appState.viewStart.toFixed(1);
    refreshPlot();
    updateProgressBar();
}

function updateViewWindow() {
    stopReplay();
    const val = parseFloat(document.getElementById('view-window').value);
    if (val > 0 && val <= appState.totalDuration) {
        appState.viewWindow = val;
        refreshPlot();
    }
}

function jumpToTime() {
    stopReplay();
    const val = parseFloat(document.getElementById('view-start').value);
    if (!isNaN(val) && val >= 0 && val < appState.totalDuration) {
        appState.viewStart = val;
        refreshPlot();
        updateProgressBar();
    }
}

// ── Preprocessing Actions ────────────────────────────────────
async function applyBandpass() {
    showLoading('Applying bandpass filter...');
    try {
        const l = parseFloat(document.getElementById('bp-low').value);
        const h = parseFloat(document.getElementById('bp-high').value);
        const res = await api('/api/preprocess', 'POST', {
            action: 'bandpass', l_freq: l, h_freq: h
        });
        updateHistoryPanel(res.history);
        await refreshPlot();
        updateWorkflowBreadcrumb('processed');
        showToast(`Bandpass ${l}-${h} Hz applied`, 'success');
    } catch (e) {
        showToast('Bandpass error: ' + e.message, 'error');
    }
    hideLoading();
}

async function applyNotch() {
    showLoading('Applying notch filter...');
    try {
        const freq = parseFloat(document.getElementById('notch-freq').value);
        const res = await api('/api/preprocess', 'POST', {
            action: 'notch', freq: freq
        });
        updateHistoryPanel(res.history);
        await refreshPlot();
        updateWorkflowBreadcrumb('processed');
        showToast(`Notch ${freq} Hz applied`, 'success');
    } catch (e) {
        showToast('Notch error: ' + e.message, 'error');
    }
    hideLoading();
}

async function applyNormalize() {
    showLoading('Normalizing...');
    try {
        const normalizeScale = parseFloat(document.getElementById('normalize-scale').value);
        const res = await api('/api/preprocess', 'POST', {
            action: 'normalize',
            normalize_scale: normalizeScale,
        });
        updateHistoryPanel(res.history);
        await refreshPlot();
        updateWorkflowBreadcrumb('processed');
        showToast(`Robust normalization applied (gain=${normalizeScale})`, 'success');
    } catch (e) {
        showToast('Normalize error: ' + e.message, 'error');
    }
    hideLoading();
}

async function resetData() {
    showLoading('Resetting to raw data...');
    try {
        const res = await api('/api/preprocess', 'POST', { action: 'reset' });
        updateHistoryPanel(res.history);
        await refreshPlot();
        updateWorkflowBreadcrumb('loaded');
        showToast('Reset to raw data', 'info');
    } catch (e) {
        showToast('Reset error: ' + e.message, 'error');
    }
    hideLoading();
}

// ── PSD ──────────────────────────────────────────────────────
async function loadPSD() {
    if (!appState.loaded) return;
    showLoading('Computing PSD...');
    try {
        const data = await api('/api/psd');
        plotPSD(data);
        showToast('PSD computed', 'success');
    } catch (e) {
        showToast('PSD error: ' + e.message, 'error');
    }
    hideLoading();
}

function plotPSD(data) {
    const container = document.getElementById('psd-chart');
    const traces = [];

    data.ch_names.forEach((ch, i) => {
        if (!appState.visibleChannels[ch]) return;
        // Convert to dB
        const psdDb = data.psd[i].map(v => 10 * Math.log10(v + 1e-20));
        traces.push({
            x: data.freqs,
            y: psdDb,
            type: 'scatter',
            mode: 'lines',
            name: ch,
            line: { color: CHANNEL_COLORS[i % 16], width: 1.5 },
        });
    });

    const layout = {
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: 'Power Spectral Density (Welch)', font: { size: 13, color: '#475569' } },
        xaxis: {
            ...PLOTLY_LAYOUT_DEFAULTS.xaxis,
            title: { text: 'Frequency (Hz)', font: { size: 11, color: '#64748b' } },
            range: [0, 60],
        },
        yaxis: {
            ...PLOTLY_LAYOUT_DEFAULTS.yaxis,
            title: { text: 'Power (dB)', font: { size: 11, color: '#64748b' } },
        },
        showlegend: true,
        height: 400,
        legend: { ...PLOTLY_LAYOUT_DEFAULTS.legend, x: 1.01, y: 1 },
    };

    // Add EEG band annotations
    const bands = [
        { name: 'δ', range: [0.5, 4], color: 'rgba(99,102,241,0.06)' },
        { name: 'θ', range: [4, 8], color: 'rgba(139,92,246,0.06)' },
        { name: 'α', range: [8, 13], color: 'rgba(5,150,105,0.06)' },
        { name: 'β', range: [13, 30], color: 'rgba(217,119,6,0.06)' },
        { name: 'γ', range: [30, 45], color: 'rgba(234,88,12,0.06)' },
    ];

    layout.shapes = bands.map(b => ({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: b.range[0], x1: b.range[1], y0: 0, y1: 1,
        fillcolor: b.color, line: { width: 0 },
    }));

    layout.annotations = (layout.annotations || []).concat(bands.map(b => ({
        x: (b.range[0] + b.range[1]) / 2,
        y: 1.05,
        xref: 'x', yref: 'paper',
        text: b.name,
        showarrow: false,
        font: { size: 12, color: '#94a3b8' },
    })));

    Plotly.react(container, traces, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        scrollZoom: true,
    });
}

// ── Artifacts ────────────────────────────────────────────────
async function detectArtifacts() {
    if (!appState.loaded) return;
    showLoading('Detecting artifacts...');
    try {
        const maxAbs = parseFloat(document.getElementById('artifact-thresh').value);
        const data = await api(`/api/artifacts?max_abs_uv=${maxAbs}`);
        displayArtifacts(data);
        showToast(`Found ${data.n_bad}/${data.n_total} bad segments`, data.n_bad > 0 ? 'warning' : 'success');
    } catch (e) {
        showToast('Artifact detection error: ' + e.message, 'error');
    }
    hideLoading();
}

function displayArtifacts(data) {
    const container = document.getElementById('artifact-results');
    container.innerHTML = `
        <div class="artifact-summary">
            <div class="artifact-stat good">
                <div class="value">${data.n_total - data.n_bad}</div>
                <div class="label">Good Segments</div>
            </div>
            <div class="artifact-stat bad">
                <div class="value">${data.n_bad}</div>
                <div class="label">Bad Segments</div>
            </div>
        </div>
        <div style="font-size:0.76rem;color:var(--text-muted);margin-bottom:8px;">
            ${data.pct_bad}% rejected
        </div>
    `;

    if (data.bad_segments.length > 0) {
        let html = '<div style="max-height:120px;overflow-y:auto;">';
        data.bad_segments.slice(0, 20).forEach(seg => {
            html += `<div class="history-item" style="cursor:pointer" onclick="appState.viewStart=${seg.start_sec};document.getElementById('view-start').value='${seg.start_sec}';refreshPlot();">
                ${seg.start_sec}s–${seg.end_sec}s: ${seg.reason}
            </div>`;
        });
        if (data.bad_segments.length > 20) {
            html += `<div class="history-item">...and ${data.bad_segments.length - 20} more</div>`;
        }
        html += '</div>';
        container.innerHTML += html;
    }
}

// ── Statistics ───────────────────────────────────────────────
async function loadStatistics() {
    if (!appState.loaded) return;
    try {
        const data = await api('/api/statistics');
        displayStatistics(data.statistics);
    } catch (e) {
        showToast('Statistics error: ' + e.message, 'error');
    }
}

function displayStatistics(stats) {
    const tbody = document.getElementById('stats-body');
    tbody.innerHTML = '';
    stats.forEach((s, i) => {
        tbody.innerHTML += `
            <tr>
                <td style="color:${CHANNEL_COLORS[i % 16]};font-weight:600;">${s.channel}</td>
                <td>${s.mean.toFixed(2)}</td>
                <td>${s.std.toFixed(2)}</td>
                <td>${s.min.toFixed(2)}</td>
                <td>${s.max.toFixed(2)}</td>
                <td>${s.rms.toFixed(2)}</td>
            </tr>
        `;
    });
}

// ── Export ────────────────────────────────────────────────────
async function exportData() {
    if (!appState.loaded) return;
    showLoading('Exporting data...');
    try {
        const data = await api('/api/export', 'POST');
        showToast(`Exported to ${data.filename}`, 'success');
    } catch (e) {
        showToast('Export error: ' + e.message, 'error');
    }
    hideLoading();
}

// ── Live Stream ─────────────────────────────────────────────
function setLiveStatus(text, type = 'info') {
    const el = document.getElementById('live-status');
    if (!el) return;
    const colors = { info: 'var(--text-muted)', success: '#059669', error: '#dc2626' };
    el.style.color = colors[type] || colors.info;
    el.textContent = text;
}

function startLivePolling() {
    stopLivePolling();
    appState.livePollTimer = setInterval(async () => {
        if (!appState.liveRunning) return;
        await refreshPlot();
    }, 350);
}

function stopLivePolling() {
    if (appState.livePollTimer) {
        clearInterval(appState.livePollTimer);
        appState.livePollTimer = null;
    }
}

async function refreshLiveStatus() {
    try {
        const res = await api('/api/live/status');
        appState.liveRunning = !!res.running;
        const serialInput = document.getElementById('live-serial-port');
        if (serialInput) {
            if (res.config && res.config.serial_port) {
                serialInput.value = res.config.serial_port;
            } else if (!serialInput.value) {
                serialInput.value = navigator.platform?.toLowerCase().includes('win') ? 'COM3' : '/dev/ttyUSB0';
            }
        }
        if (res.running) {
            setLiveStatus(`Streaming (${res.n_channels} ch @ ${res.sfreq} Hz, buffer ${res.n_samples_buffered} samples)`, 'success');
        } else if (res.error) {
            setLiveStatus(`Error: ${res.error}`, 'error');
        } else {
            setLiveStatus('Idle', 'info');
        }
    } catch (e) {
        setLiveStatus(`Status error: ${e.message}`, 'error');
    }
}

async function startLiveStream() {
    stopReplay();
    showLoading('Starting live stream...');
    try {
        const serialPort = document.getElementById('live-serial-port').value.trim();
        const boardId = parseInt(document.getElementById('live-board-id').value, 10);
        const lFreq = parseFloat(document.getElementById('live-bp-low').value);
        const hFreq = parseFloat(document.getElementById('live-bp-high').value);
        const notch = parseFloat(document.getElementById('live-notch').value);

        const res = await api('/api/live/start', 'POST', {
            serial_port: serialPort,
            board_id: boardId,
            l_freq: lFreq,
            h_freq: hFreq,
            notch: notch,
            order: 4,
            buffer_seconds: 30,
        });

        appState.loaded = true;
        appState.liveRunning = true;
        appState.channelNames = res.info.ch_names || [];
        appState.visibleChannels = {};
        appState.channelNames.forEach(ch => appState.visibleChannels[ch] = true);
        appState.sfreq = res.info.sfreq || appState.sfreq;
        appState.totalDuration = 0;
        appState.viewWindow = 10;
        appState.viewStart = 0;

        buildChannelToggles(appState.channelNames);
        updateInfoPanel({
            n_channels: appState.channelNames.length,
            n_samples: 0,
            sfreq: appState.sfreq,
            duration_sec: 0,
            source_format: 'live_stream',
        });
        updateHistoryPanel(['Live stream starting...']);
        enableControls(true);
        document.getElementById('status-text').textContent = 'Live Streaming';
        setLiveStatus(`Streaming (${appState.channelNames.length} ch @ ${appState.sfreq} Hz)`, 'success');

        startLivePolling();
        await refreshPlot();
        showToast('Live stream started', 'success');
    } catch (e) {
        appState.liveRunning = false;
        stopLivePolling();
        setLiveStatus(`Error: ${e.message}`, 'error');
        showToast('Live start error: ' + e.message, 'error');
    }
    hideLoading();
}

async function applySmoothing() {
    showLoading('Applying smoothing...');
    try {
        const windowSamples = parseInt(document.getElementById('smooth-window').value, 10);
        const res = await api('/api/preprocess', 'POST', {
            action: 'smooth',
            window_samples: windowSamples,
        });
        updateHistoryPanel(res.history);
        await refreshPlot();
        showToast(`Smoothing applied (window=${windowSamples})`, 'success');
    } catch (e) {
        showToast('Smoothing error: ' + e.message, 'error');
    }
    hideLoading();
}

async function stopLiveStream() {
    showLoading('Stopping live stream...');
    try {
        await api('/api/live/stop', 'POST', {});
        appState.liveRunning = false;
        stopLivePolling();
        document.getElementById('status-text').textContent = appState.loaded ? 'Data Loaded' : 'Ready';
        setLiveStatus('Idle', 'info');
        showToast('Live stream stopped', 'info');
    } catch (e) {
        setLiveStatus(`Stop error: ${e.message}`, 'error');
        showToast('Live stop error: ' + e.message, 'error');
    }
    hideLoading();
}

// ── UI Helpers ───────────────────────────────────────────────
function updateHistoryPanel(history) {
    const container = document.getElementById('history-list');
    container.innerHTML = history.map(h =>
        `<div class="history-item">${h}</div>`
    ).join('');
    container.scrollTop = container.scrollHeight;
}

function enableControls(enabled) {
    document.querySelectorAll('.requires-data').forEach(el => {
        el.disabled = !enabled;
    });
}

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabGroup = btn.closest('.card').querySelectorAll('.tab-btn');
            const contents = btn.closest('.card').querySelectorAll('.tab-content');

            tabGroup.forEach(b => b.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.classList.add('active');

            // Auto-load data for tabs
            if (btn.dataset.tab === 'tab-psd') loadPSD();
            if (btn.dataset.tab === 'tab-stats') loadStatistics();
            if (btn.dataset.tab === 'tab-bandpower') loadBandPower();
            if (btn.dataset.tab === 'tab-topomap') loadTopomap();
        });
    });
}

function setupEventListeners() {
    // View window changes
    document.getElementById('view-window').addEventListener('change', updateViewWindow);
    document.getElementById('view-start').addEventListener('change', jumpToTime);
    document.getElementById('replay-speed')?.addEventListener('change', setReplaySpeed);

    // Select/deselect all channels
    document.getElementById('btn-select-all')?.addEventListener('click', () => {
        appState.channelNames.forEach(ch => appState.visibleChannels[ch] = true);
        document.querySelectorAll('#channel-grid input').forEach(cb => cb.checked = true);
        refreshPlot();
    });

    document.getElementById('btn-deselect-all')?.addEventListener('click', () => {
        appState.channelNames.forEach(ch => appState.visibleChannels[ch] = false);
        document.querySelectorAll('#channel-grid input').forEach(cb => cb.checked = false);
        refreshPlot();
    });
}

// ── Dark Mode ──────────────────────────────────────────────
function toggleDarkMode() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('eeg-dark-mode', isDark ? '1' : '0');
    const btn = document.getElementById('btn-dark-mode');
    if (btn) btn.textContent = isDark ? '☀️' : '🌙';
    // Replot with correct theme
    if (appState.loaded) refreshPlot();
}

function initDarkMode() {
    if (localStorage.getItem('eeg-dark-mode') === '1') {
        document.body.classList.add('dark');
        const btn = document.getElementById('btn-dark-mode');
        if (btn) btn.textContent = '☀️';
    }
}

// ── Resample ───────────────────────────────────────────────
async function applyResample() {
    showLoading('Resampling...');
    try {
        const targetFreq = parseFloat(document.getElementById('resample-freq').value);
        const res = await api('/api/preprocess', 'POST', {
            action: 'resample',
            target_sfreq: targetFreq,
        });
        updateHistoryPanel(res.history);
        await refreshPlot();
        updateWorkflowBreadcrumb('processed');
        showToast(`Resampled to ${targetFreq} Hz`, 'success');
    } catch (e) {
        showToast('Resample error: ' + e.message, 'error');
    }
    hideLoading();
}

// ── Band Power ─────────────────────────────────────────────
async function loadBandPower() {
    if (!appState.loaded) return;
    try {
        const data = await api('/api/bandpower');
        const container = document.getElementById('bandpower-chart');
        const isDark = document.body.classList.contains('dark');

        const bandNames = data.band_names;
        const bandColors = {
            delta: '#6366f1', theta: '#8b5cf6',
            alpha: '#059669', beta: '#d97706', gamma: '#dc2626'
        };

        const traces = bandNames.map(band => ({
            name: band.charAt(0).toUpperCase() + band.slice(1),
            x: data.ch_names,
            y: data.bands[band],
            type: 'bar',
            marker: { color: bandColors[band] || '#6366f1' },
        }));

        const layout = {
            title: 'EEG Band Power by Channel',
            barmode: 'group',
            xaxis: { title: 'Channel' },
            yaxis: { title: 'Power (µV²/Hz)' },
            paper_bgcolor: isDark ? '#1e293b' : '#ffffff',
            plot_bgcolor: isDark ? '#0f172a' : '#f8fafc',
            font: { color: isDark ? '#e2e8f0' : '#1e293b', family: 'Inter' },
            margin: { t: 40, r: 20, b: 60, l: 60 },
            legend: { orientation: 'h', y: -0.2 },
        };

        Plotly.newPlot(container, traces, layout, { responsive: true });
    } catch (e) {
        showToast('Band power error: ' + e.message, 'error');
    }
}

// ── ICA ────────────────────────────────────────────────────
async function runICA() {
    if (!appState.loaded) return;
    showLoading('Decomposing with ICA...');
    try {
        const nComp = parseInt(document.getElementById('ica-n-components').value, 10) || 0;
        const data = await api('/api/ica', 'POST', { n_components: nComp });

        if (data.action === 'decomposed') {
            const container = document.getElementById('ica-results');
            const maxRMS = Math.max(...data.component_rms);

            let html = `<div style="font-size:0.76rem;color:var(--text-secondary);margin-top:6px;">Select components to remove:</div>`;
            html += '<div class="ica-component-list">';
            data.component_rms.forEach((rms, i) => {
                const pct = maxRMS > 0 ? (rms / maxRMS * 100) : 0;
                html += `<div class="ica-component-item">`;
                html += `<input type="checkbox" id="ica-comp-${i}" value="${i}">`;
                html += `<span>IC${i}</span>`;
                html += `<div class="ica-rms-bar"><div class="ica-rms-fill" style="width:${pct}%"></div></div>`;
                html += `<span>${rms.toFixed(2)}</span>`;
                html += `</div>`;
            });
            html += '</div>';
            html += '<button class="btn btn-danger" style="margin-top:8px;width:100%;" onclick="removeICAComponents()">Remove Selected</button>';
            container.innerHTML = html;
            showToast(`ICA decomposed into ${data.n_components} components`, 'success');
        }
    } catch (e) {
        showToast('ICA error: ' + e.message, 'error');
    }
    hideLoading();
}

async function removeICAComponents() {
    const checkboxes = document.querySelectorAll('#ica-results input[type="checkbox"]:checked');
    const exclude = Array.from(checkboxes).map(cb => parseInt(cb.value, 10));
    if (exclude.length === 0) {
        showToast('Select at least one component to remove', 'info');
        return;
    }
    showLoading('Removing ICA components...');
    try {
        const data = await api('/api/ica', 'POST', { exclude });
        document.getElementById('ica-results').innerHTML = `<div style="font-size:0.76rem;color:var(--success);margin-top:6px;">✓ Removed components ${exclude.join(', ')}</div>`;
        await refreshPlot();
        showToast(`ICA: removed ${exclude.length} components`, 'success');
    } catch (e) {
        showToast('ICA removal error: ' + e.message, 'error');
    }
    hideLoading();
}

// ── Topomap ────────────────────────────────────────────────
async function loadTopomap() {
    if (!appState.loaded) return;
    try {
        const data = await api('/api/topomap');
        const container = document.getElementById('topomap-chart');
        const isDark = document.body.classList.contains('dark');
        const channels = data.channels;

        const xs = channels.map(c => c.x);
        const ys = channels.map(c => c.y);
        const powers = channels.map(c => c.power);
        const names = channels.map(c => c.name);
        const maxPow = Math.max(...powers);
        const sizes = powers.map(p => 15 + (p / maxPow) * 35);

        // Scalp outline
        const thetaHead = Array.from({ length: 101 }, (_, i) => i * 2 * Math.PI / 100);
        const headX = thetaHead.map(t => Math.cos(t));
        const headY = thetaHead.map(t => Math.sin(t));

        // Nose indicator
        const noseX = [-0.1, 0, 0.1];
        const noseY = [1.0, 1.12, 1.0];

        const traces = [
            {
                x: headX, y: headY, type: 'scatter', mode: 'lines',
                line: { color: isDark ? '#64748b' : '#94a3b8', width: 2 },
                showlegend: false, hoverinfo: 'none'
            },
            {
                x: noseX, y: noseY, type: 'scatter', mode: 'lines',
                line: { color: isDark ? '#64748b' : '#94a3b8', width: 2 },
                showlegend: false, hoverinfo: 'none'
            },
            {
                x: xs, y: ys, type: 'scatter', mode: 'markers+text',
                text: names,
                textposition: 'top center',
                textfont: { size: 10, color: isDark ? '#e2e8f0' : '#1e293b' },
                marker: {
                    size: sizes,
                    color: powers,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: { title: 'RMS (µV)', len: 0.6 },
                    line: { width: 1, color: isDark ? '#334155' : '#e2e8f0' },
                },
                hovertemplate: '%{text}<br>RMS: %{marker.color:.2f} µV<extra></extra>',
            }
        ];

        const layout = {
            title: 'Topographic Map (10-20 System)',
            xaxis: { range: [-1.3, 1.3], showgrid: false, zeroline: false, showticklabels: false },
            yaxis: { range: [-1.3, 1.3], showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' },
            paper_bgcolor: isDark ? '#1e293b' : '#ffffff',
            plot_bgcolor: isDark ? '#1e293b' : '#ffffff',
            font: { color: isDark ? '#e2e8f0' : '#1e293b', family: 'Inter' },
            margin: { t: 40, r: 20, b: 20, l: 20 },
        };

        Plotly.newPlot(container, traces, layout, { responsive: true });
    } catch (e) {
        showToast('Topomap error: ' + e.message, 'error');
    }
}

// ── Recording ──────────────────────────────────────────────
async function startRecording() {
    if (!appState.loaded) return;
    showLoading('Starting recording...');
    try {
        const data = await api('/api/record/start', 'POST');
        document.getElementById('recording-status').textContent = `Recording: ${data.filename} (${data.samples_written} samples)`;
        document.getElementById('recording-status').style.color = '#dc2626';
        showToast('Recording started: ' + data.filename, 'success');
    } catch (e) {
        showToast('Record error: ' + e.message, 'error');
    }
    hideLoading();
}

async function stopRecording() {
    showLoading('Stopping recording...');
    try {
        const data = await api('/api/record/stop', 'POST');
        document.getElementById('recording-status').textContent = `Saved ${data.samples_written} samples to ${data.filepath || 'file'}`;
        document.getElementById('recording-status').style.color = 'var(--text-muted)';
        showToast('Recording stopped', 'info');
    } catch (e) {
        showToast('Stop record error: ' + e.message, 'error');
    }
    hideLoading();
}

// ── Collapsible Cards ──────────────────────────────────────
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    if (!card) return;
    card.classList.toggle('collapsed');
}

// ── Accordion Sections ─────────────────────────────────────
function toggleAccordion(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    section.classList.toggle('collapsed');
}

// ── Workflow Breadcrumb ────────────────────────────────────
function updateWorkflowBreadcrumb(stage) {
    appState.workflowStage = stage;
    const stages = ['ready', 'loaded', 'processed', 'analyzed'];
    const currentIdx = stages.indexOf(stage);

    stages.forEach((s, i) => {
        const el = document.getElementById('step-' + s);
        if (!el) return;
        el.classList.remove('active', 'completed');
        if (i < currentIdx) {
            el.classList.add('completed');
        } else if (i === currentIdx) {
            el.classList.add('active');
        }
    });
}

// ── Progress Bar ───────────────────────────────────────────
function updateProgressBar() {
    const fill = document.getElementById('progress-fill');
    if (!fill || appState.totalDuration <= 0) return;
    const pct = Math.min(100, ((appState.viewStart + appState.viewWindow) / appState.totalDuration) * 100);
    fill.style.width = pct + '%';
}

// Progress track click-to-seek
document.addEventListener('click', (e) => {
    const track = e.target.closest('#progress-track');
    if (!track || !appState.loaded) return;
    const rect = track.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const targetTime = pct * appState.totalDuration;
    appState.viewStart = Math.max(0, Math.min(targetTime - appState.viewWindow / 2, appState.totalDuration - appState.viewWindow));
    document.getElementById('view-start').value = appState.viewStart.toFixed(1);
    refreshPlot();
    updateProgressBar();
});

// ── Keyboard Shortcuts ─────────────────────────────────────
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't fire shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                if (appState.loaded && !appState.liveRunning) toggleReplay();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                navigate('prev');
                break;
            case 'ArrowRight':
                e.preventDefault();
                navigate('next');
                break;
            case 'Home':
                e.preventDefault();
                navigate('start');
                break;
            case 'End':
                e.preventDefault();
                navigate('end');
                break;
            case 'KeyD':
                if (!e.ctrlKey && !e.metaKey) toggleDarkMode();
                break;
        }
    });
}
