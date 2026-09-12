/**
 * VoiceGuard — Interactive Audio Upload & Real-Time Call Risk Analysis
 * Vanilla JS with 60fps Canvas Waveform, HUD Gauges & Real-time Playhead Sync
 */

(function () {
  'use strict';

  // --- State Management ---
  let analysisData = null;
  let isPlaying = false;
  let activeChunkIndex = -1;
  let animFrameId = null;

  // --- DOM Element References ---
  const dropZone = document.getElementById('dropZone');
  const audioFileInput = document.getElementById('audioFileInput');
  const analysisLoading = document.getElementById('analysisLoading');
  const loadingStepTitle = document.getElementById('loadingStepTitle');
  const loadingStepDesc = document.getElementById('loadingStepDesc');
  const analysisDashboard = document.getElementById('analysisDashboard');

  // Player Elements
  const audio = document.getElementById('audioPlayer');
  const playPauseBtn = document.getElementById('playPauseBtn');
  const playIcon = document.getElementById('playIcon');
  const pauseIcon = document.getElementById('pauseIcon');
  const restartBtn = document.getElementById('restartBtn');
  const muteBtn = document.getElementById('muteBtn');
  const volHighIcon = document.getElementById('volHighIcon');
  const volMuteIcon = document.getElementById('volMuteIcon');
  const currentTimeDisplay = document.getElementById('currentTimeDisplay');
  const totalTimeDisplay = document.getElementById('totalTimeDisplay');
  const activeFileName = document.getElementById('activeFileName');
  const audioDurationPill = document.getElementById('audioDurationPill');
  const sampleRatePill = document.getElementById('sampleRatePill');
  const dangerousSegCountPill = document.getElementById('dangerousSegCountPill');

  // Waveform Elements
  const waveformContainer = document.getElementById('waveformContainer');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const dangerOverlaysLayer = document.getElementById('dangerOverlaysLayer');
  const playheadCursor = document.getElementById('playheadCursor');
  const waveformTooltip = document.getElementById('waveformTooltip');
  const ctx = waveformCanvas.getContext('2d');

  // Threat Alert Banner Elements
  const threatAlertBanner = document.getElementById('threatAlertBanner');
  const alertIconBox = document.getElementById('alertIconBox');
  const alertBannerTag = document.getElementById('alertBannerTag');
  const alertBannerText = document.getElementById('alertBannerText');
  const alertTimestamp = document.getElementById('alertTimestamp');

  // HUD Elements
  const liveRiskValue = document.getElementById('liveRiskValue');
  const liveRiskGaugeCircle = document.getElementById('liveRiskGaugeCircle');
  const liveRiskStatusBadge = document.getElementById('liveRiskStatusBadge');
  const liveAcousticVal = document.getElementById('liveAcousticVal');
  const liveAcousticVerdict = document.getElementById('liveAcousticVerdict');
  const liveAcousticFill = document.getElementById('liveAcousticFill');
  const hudJitterVal = document.getElementById('hudJitterVal');
  const hudPhaseVal = document.getElementById('hudPhaseVal');
  const liveContextVal = document.getElementById('liveContextVal');
  const liveContextVerdict = document.getElementById('liveContextVerdict');
  const liveContextFill = document.getElementById('liveContextFill');
  const hudCredVal = document.getElementById('hudCredVal');
  const hudUrgencyVal = document.getElementById('hudUrgencyVal');

  // Panes Elements
  const transcriptContainer = document.getElementById('transcriptContainer');
  const dangerousSegmentsContainer = document.getElementById('dangerousSegmentsContainer');
  const flaggedCountBadge = document.getElementById('flaggedCountBadge');

  // Conclusion Card Elements
  const conclusionBanner = document.getElementById('conclusionBanner');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictTitle = document.getElementById('verdictTitle');
  const concPeakRisk = document.getElementById('concPeakRisk');
  const concAcousticScore = document.getElementById('concAcousticScore');
  const concDangerCount = document.getElementById('concDangerCount');
  const concSafeRatio = document.getElementById('concSafeRatio');
  const concExecutiveSummary = document.getElementById('concExecutiveSummary');
  const specJitter = document.getElementById('specJitter');
  const specShimmer = document.getElementById('specShimmer');
  const specPhase = document.getElementById('specPhase');
  const specProb = document.getElementById('specProb');
  const threatsBadgesContainer = document.getElementById('threatsBadgesContainer');
  const recommendationsList = document.getElementById('recommendationsList');

  // 1-Click Demo Buttons
  const sampleCards = document.querySelectorAll('.sample-card');

  // Circumference for radial gauge: 2 * Math.PI * 68 = ~427.25
  const GAUGE_CIRCUMFERENCE = 427.25;

  // ---------------------------------------------------------------------------
  // Time Formatter (MM:SS)
  // ---------------------------------------------------------------------------
  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  // ---------------------------------------------------------------------------
  // Audio Playback Controls
  // ---------------------------------------------------------------------------
  function togglePlay() {
    if (!audio.src) return;
    if (audio.paused) {
      audio.play().then(() => {
        isPlaying = true;
        updatePlayIcons();
        startSyncLoop();
      }).catch(err => {
        console.warn('Playback blocked:', err);
      });
    } else {
      audio.pause();
      isPlaying = false;
      updatePlayIcons();
      stopSyncLoop();
    }
  }

  function updatePlayIcons() {
    if (audio.paused) {
      playIcon.style.display = 'block';
      pauseIcon.style.display = 'none';
    } else {
      playIcon.style.display = 'none';
      pauseIcon.style.display = 'block';
    }
  }

  playPauseBtn.addEventListener('click', togglePlay);

  restartBtn.addEventListener('click', () => {
    if (!audio.src) return;
    audio.currentTime = 0;
    audio.play();
    isPlaying = true;
    updatePlayIcons();
    startSyncLoop();
  });

  muteBtn.addEventListener('click', () => {
    audio.muted = !audio.muted;
    if (audio.muted) {
      volHighIcon.style.display = 'none';
      volMuteIcon.style.display = 'block';
    } else {
      volHighIcon.style.display = 'block';
      volMuteIcon.style.display = 'none';
    }
  });

  // Speed selector
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      audio.playbackRate = parseFloat(btn.getAttribute('data-speed'));
    });
  });

  audio.addEventListener('ended', () => {
    isPlaying = false;
    updatePlayIcons();
    stopSyncLoop();
  });

  // ---------------------------------------------------------------------------
  // Canvas Waveform Drawing & Danger Zones
  // ---------------------------------------------------------------------------
  function resizeCanvas() {
    const rect = waveformContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    waveformCanvas.width = rect.width * dpr;
    waveformCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    renderWaveform();
  }

  window.addEventListener('resize', () => {
    if (analysisData) resizeCanvas();
  });

  function renderWaveform() {
    if (!analysisData || !analysisData.waveform_peaks) return;

    const width = waveformContainer.clientWidth;
    const height = waveformContainer.clientHeight;
    ctx.clearRect(0, 0, width, height);

    const peaks = analysisData.waveform_peaks;
    const numBars = peaks.length;
    const barWidth = Math.max(2, (width / numBars) - 2);
    const centerY = height / 2;

    const currentTime = audio.currentTime || 0;
    const duration = analysisData.duration_sec || 1;
    const progressRatio = Math.min(1.0, currentTime / duration);
    const activeBarIndex = Math.floor(progressRatio * numBars);

    // Draw baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Draw bars
    for (let i = 0; i < numBars; i++) {
      const x = i * (width / numBars);
      const barHeight = Math.max(4, peaks[i] * (height * 0.8));
      const topY = centerY - (barHeight / 2);

      // Check if bar is in a dangerous segment
      const barSec = (i / numBars) * duration;
      const isDangerous = analysisData.dangerous_segments.some(
        seg => barSec >= seg.start_sec && barSec <= seg.end_sec
      );

      if (i <= activeBarIndex) {
        // Played portion
        if (isDangerous) {
          ctx.fillStyle = '#ef4444'; // Crimson danger
        } else {
          ctx.fillStyle = '#06b6d4'; // Cyan active
        }
      } else {
        // Unplayed portion
        if (isDangerous) {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.45)'; // Dimmed crimson
        } else {
          ctx.fillStyle = 'rgba(148, 163, 184, 0.35)'; // Slate unplayed
        }
      }

      ctx.beginPath();
      ctx.roundRect(x, topY, barWidth, barHeight, 2);
      ctx.fill();
    }
  }

  function setupDangerBands() {
    dangerOverlaysLayer.innerHTML = '';
    if (!analysisData || !analysisData.dangerous_segments) return;

    const duration = analysisData.duration_sec || 1;
    analysisData.dangerous_segments.forEach(seg => {
      const leftPct = (seg.start_sec / duration) * 100;
      const widthPct = ((seg.end_sec - seg.start_sec) / duration) * 100;

      const band = document.createElement('div');
      band.className = 'danger-band';
      band.style.left = `${leftPct}%`;
      band.style.width = `${Math.max(1.5, widthPct)}%`;
      band.title = `Threat: ${seg.primary_threat} (${seg.peak_risk}% risk)`;
      dangerOverlaysLayer.appendChild(band);
    });
  }

  // Waveform Click Seeking & Tooltip Hover
  waveformContainer.addEventListener('click', (e) => {
    if (!analysisData || !audio.duration) return;
    const rect = waveformContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    audio.currentTime = ratio * analysisData.duration_sec;
    renderWaveform();
    updateLiveState(audio.currentTime);
    if (audio.paused) {
      audio.play();
      isPlaying = true;
      updatePlayIcons();
      startSyncLoop();
    }
  });

  waveformContainer.addEventListener('mousemove', (e) => {
    if (!analysisData) return;
    const rect = waveformContainer.getBoundingClientRect();
    const hoverX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, hoverX / rect.width));
    const hoverTime = ratio * analysisData.duration_sec;

    waveformTooltip.style.display = 'block';
    waveformTooltip.style.left = `${Math.min(rect.width - 50, hoverX + 10)}px`;
    waveformTooltip.textContent = formatTime(hoverTime);
  });

  waveformContainer.addEventListener('mouseleave', () => {
    waveformTooltip.style.display = 'none';
  });

  // ---------------------------------------------------------------------------
  // 60FPS Playback Synchronization Loop
  // ---------------------------------------------------------------------------
  function startSyncLoop() {
    if (animFrameId) cancelAnimationFrame(animFrameId);

    function loop() {
      if (!audio.paused && analysisData) {
        const curTime = audio.currentTime;
        currentTimeDisplay.textContent = formatTime(curTime);

        // Update playhead cursor position
        const progressPct = (curTime / (analysisData.duration_sec || 1)) * 100;
        playheadCursor.style.left = `${progressPct}%`;

        // Update active waveform coloring
        renderWaveform();

        // Update HUD & Threat alerts
        updateLiveState(curTime);
      }
      animFrameId = requestAnimationFrame(loop);
    }
    animFrameId = requestAnimationFrame(loop);
  }

  function stopSyncLoop() {
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Real-time State & HUD Syncing
  // ---------------------------------------------------------------------------
  function updateLiveState(currentTime) {
    if (!analysisData || !analysisData.chunks) return;

    // Find active chunk
    const chunks = analysisData.chunks;
    const activeChunk = chunks.find(c => currentTime >= c.start_sec && currentTime <= c.end_sec) || chunks[chunks.length - 1];

    if (!activeChunk) return;

    const riskScore = activeChunk.risk_score;
    const acousticScore = activeChunk.acoustic_score;
    const contextScore = activeChunk.context_score;

    // 1. Update Radial Composite Risk Gauge
    liveRiskValue.textContent = Math.round(riskScore);
    const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * (riskScore / 100));
    liveRiskGaugeCircle.style.strokeDashoffset = offset;

    if (riskScore >= 75) {
      liveRiskGaugeCircle.style.stroke = 'var(--danger-crimson)';
      liveRiskStatusBadge.className = 'hud-status-badge critical';
      liveRiskStatusBadge.textContent = '🚨 CRITICAL RISK';
    } else if (riskScore >= 50) {
      liveRiskGaugeCircle.style.stroke = 'var(--warning-amber)';
      liveRiskStatusBadge.className = 'hud-status-badge warning';
      liveRiskStatusBadge.textContent = '⚠️ SUSPICIOUS PATTERN';
    } else {
      liveRiskGaugeCircle.style.stroke = 'var(--safe-emerald)';
      liveRiskStatusBadge.className = 'hud-status-badge safe';
      liveRiskStatusBadge.textContent = 'SAFE RANGE';
    }

    // 2. Update Acoustic Meter
    liveAcousticVal.textContent = `${acousticScore}%`;
    liveAcousticFill.style.width = `${acousticScore}%`;
    if (acousticScore >= 70) {
      liveAcousticVerdict.textContent = '🚨 Synthetic Voice Artifacts';
      liveAcousticVerdict.style.color = 'var(--danger-crimson)';
      hudJitterVal.textContent = 'Abnormally Low (0.005%)';
      hudJitterVal.style.color = 'var(--danger-crimson)';
      hudPhaseVal.textContent = 'High Coherence (Vocoder)';
    } else if (acousticScore >= 50) {
      liveAcousticVerdict.textContent = '⚠️ Elevated Spoof Probability';
      liveAcousticVerdict.style.color = 'var(--warning-amber)';
      hudJitterVal.textContent = 'Low Jitter (0.011%)';
      hudPhaseVal.textContent = 'Moderate Coherence';
    } else {
      liveAcousticVerdict.textContent = 'Natural Prosody';
      liveAcousticVerdict.style.color = 'var(--safe-emerald)';
      hudJitterVal.textContent = 'Normal Range (2.2%)';
      hudJitterVal.style.color = 'var(--text-white)';
      hudPhaseVal.textContent = 'Natural Entropy';
    }

    // 3. Update Context / Scam Meter
    liveContextVal.textContent = `${contextScore}%`;
    liveContextFill.style.width = `${contextScore}%`;
    if (contextScore >= 70) {
      liveContextVerdict.textContent = '🚨 Credential Harvesting';
      liveContextVerdict.style.color = 'var(--danger-crimson)';
      hudCredVal.textContent = 'Active Solicitation!';
      hudCredVal.style.color = 'var(--danger-crimson)';
      hudUrgencyVal.textContent = 'High Coercion';
    } else if (contextScore >= 40) {
      liveContextVerdict.textContent = '⚠️ Urgent Financial Demand';
      liveContextVerdict.style.color = 'var(--warning-amber)';
      hudCredVal.textContent = 'Account Questioning';
      hudUrgencyVal.textContent = 'Pressure Applied';
    } else {
      liveContextVerdict.textContent = 'No Threat Detected';
      liveContextVerdict.style.color = 'var(--safe-emerald)';
      hudCredVal.textContent = 'None';
      hudCredVal.style.color = 'var(--text-white)';
      hudUrgencyVal.textContent = 'None';
    }

    // 4. Update Threat Alert Banner
    alertTimestamp.textContent = formatTime(currentTime);

    if (activeChunk.severity === 'critical') {
      threatAlertBanner.className = 'threat-alert-banner danger-state';
      alertBannerTag.textContent = '🚨 THREAT ALERT: CRITICAL DANGER';
      alertBannerText.textContent = activeChunk.signals[0] || 'Active credential harvesting or cloned voice extortion detected!';
      alertIconBox.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
      `;
    } else if (activeChunk.severity === 'warning') {
      threatAlertBanner.className = 'threat-alert-banner warning-state';
      alertBannerTag.textContent = '⚠️ CAUTION: SUSPICIOUS PATTERN';
      alertBannerText.textContent = activeChunk.signals[0] || 'Unusual prosody or pressure tactic observed.';
      alertIconBox.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
        </svg>
      `;
    } else {
      threatAlertBanner.className = 'threat-alert-banner safe-state';
      alertBannerTag.textContent = 'PLAYHEAD STATE: SAFE';
      alertBannerText.textContent = 'Audio playing in safe zone. No active social engineering threats detected at this timestamp.';
      alertIconBox.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="M9 12l2 2 4-4"/>
        </svg>
      `;
    }

    // 5. Update Transcript Active Line
    updateTranscriptHighlight(currentTime);
  }

  // ---------------------------------------------------------------------------
  // Transcript Highlight & Click-to-Seek
  // ---------------------------------------------------------------------------
  function renderTranscript() {
    transcriptContainer.innerHTML = '';
    if (!analysisData || !analysisData.transcript_lines || analysisData.transcript_lines.length === 0) {
      transcriptContainer.innerHTML = '<div class="no-segments-msg">No transcript available for this audio.</div>';
      return;
    }

    analysisData.transcript_lines.forEach((line, idx) => {
      const lineEl = document.createElement('div');
      lineEl.className = `transcript-line ${line.risk_level}-flag`;
      lineEl.id = `transcriptLine-${idx}`;
      lineEl.setAttribute('data-start', line.start_sec);
      lineEl.setAttribute('data-end', line.end_sec);

      // Highlight flagged phrases in text
      let highlightedText = line.text;
      if (line.flagged_phrases && line.flagged_phrases.length > 0) {
        line.flagged_phrases.forEach(phrase => {
          const regex = new RegExp(`(${phrase})`, 'gi');
          highlightedText = highlightedText.replace(regex, '<span class="flagged-word">$1</span>');
        });
      }

      lineEl.innerHTML = `
        <div class="t-meta">
          <span class="t-speaker">${line.speaker}</span>
          <span class="t-time">${formatTime(line.start_sec)} - ${formatTime(line.end_sec)}</span>
        </div>
        <p class="t-text">${highlightedText}</p>
      `;

      // Click to seek
      lineEl.addEventListener('click', () => {
        audio.currentTime = line.start_sec;
        renderWaveform();
        updateLiveState(line.start_sec);
        if (audio.paused) {
          audio.play();
          isPlaying = true;
          updatePlayIcons();
          startSyncLoop();
        }
      });

      transcriptContainer.appendChild(lineEl);
    });
  }

  function updateTranscriptHighlight(currentTime) {
    const lines = document.querySelectorAll('.transcript-line');
    lines.forEach(lineEl => {
      const start = parseFloat(lineEl.getAttribute('data-start'));
      const end = parseFloat(lineEl.getAttribute('data-end'));

      if (currentTime >= start && currentTime <= end) {
        if (!lineEl.classList.contains('active')) {
          lines.forEach(l => l.classList.remove('active'));
          lineEl.classList.add('active');
          lineEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Dangerous Segments Rendering
  // ---------------------------------------------------------------------------
  function renderDangerousSegments() {
    dangerousSegmentsContainer.innerHTML = '';
    const segments = analysisData.dangerous_segments || [];
    flaggedCountBadge.textContent = `${segments.length} Flag${segments.length === 1 ? '' : 's'}`;
    dangerousSegCountPill.textContent = `${segments.length} Threat${segments.length === 1 ? '' : 's'} Flagged`;

    if (segments.length === 0) {
      dangerousSegmentsContainer.innerHTML = `
        <div class="no-segments-msg">
          ✅ No high-risk segments flagged. All time windows within safe threshold limits.
        </div>
      `;
      return;
    }

    segments.forEach(seg => {
      const item = document.createElement('div');
      item.className = 'segment-item';
      item.innerHTML = `
        <div class="seg-header">
          <span class="seg-time">⏱️ ${formatTime(seg.start_sec)} &rarr; ${formatTime(seg.end_sec)}</span>
          <span class="seg-risk">${seg.peak_risk}% RISK</span>
        </div>
        <div class="seg-threat">${seg.primary_threat}</div>
        <div class="seg-snippet">"${seg.transcript_snippet}"</div>
      `;

      // Click to jump to segment
      item.addEventListener('click', () => {
        audio.currentTime = seg.start_sec;
        renderWaveform();
        updateLiveState(seg.start_sec);
        if (audio.paused) {
          audio.play();
          isPlaying = true;
          updatePlayIcons();
          startSyncLoop();
        }
      });

      dangerousSegmentsContainer.appendChild(item);
    });
  }

  // ---------------------------------------------------------------------------
  // Conclusion Card Population
  // ---------------------------------------------------------------------------
  function renderConclusion() {
    const conc = analysisData.conclusion;
    if (!conc) return;

    // Verdict styling
    verdictTitle.textContent = conc.verdict;
    if (conc.verdict_level === 'critical') {
      conclusionBanner.className = 'conclusion-banner';
      verdictTitle.style.color = 'var(--danger-crimson)';
      verdictIcon.innerHTML = `
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--danger-crimson)" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
      `;
    } else if (conc.verdict_level === 'warning') {
      conclusionBanner.className = 'conclusion-banner warning-verdict';
      verdictTitle.style.color = 'var(--warning-amber)';
      verdictIcon.innerHTML = `
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--warning-amber)" stroke-width="2.5">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
        </svg>
      `;
    } else {
      conclusionBanner.className = 'conclusion-banner safe-verdict';
      verdictTitle.style.color = 'var(--safe-emerald)';
      verdictIcon.innerHTML = `
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--safe-emerald)" stroke-width="2.5">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="M9 12l2 2 4-4"/>
        </svg>
      `;
    }

    // 4 Metrics
    concPeakRisk.textContent = `${conc.peak_risk}%`;
    concAcousticScore.textContent = `${conc.voice_clone_probability}%`;
    concDangerCount.textContent = `${conc.dangerous_segments_count} Segments`;
    concSafeRatio.textContent = `${conc.safe_ratio_pct}%`;

    // Summary
    concExecutiveSummary.textContent = conc.executive_summary;

    // Forensic Specs
    if (conc.acoustic_summary) {
      specJitter.textContent = conc.acoustic_summary.jitter_status || 'Normal';
      specShimmer.textContent = conc.acoustic_summary.shimmer_status || 'Natural Dynamic Range';
      specPhase.textContent = conc.acoustic_summary.phase_coherence || 'Natural Phase Distribution';
      specProb.textContent = conc.acoustic_summary.voice_synthesis_likelihood || `${conc.voice_clone_probability}%`;
    }

    // Threats Tags
    threatsBadgesContainer.innerHTML = '';
    const threats = conc.detected_threats || [];
    if (threats.length === 0) {
      threatsBadgesContainer.innerHTML = '<span class="tag">No overt social engineering phrases detected</span>';
    } else {
      threats.forEach(t => {
        const badge = document.createElement('span');
        badge.className = `threat-pill ${t.tier}`;
        badge.textContent = `🚨 ${t.phrase.toUpperCase()} (${t.count}x)`;
        threatsBadgesContainer.appendChild(badge);
      });
    }

    // Recommendations
    recommendationsList.innerHTML = '';
    const recs = conc.recommendations || [];
    recs.forEach(rec => {
      const li = document.createElement('li');
      li.textContent = rec;
      recommendationsList.appendChild(li);
    });
  }

  // ---------------------------------------------------------------------------
  // Load & Display Full Analysis Data
  // ---------------------------------------------------------------------------
  function displayAnalysis(data) {
    analysisData = data;

    // Populate header info
    activeFileName.textContent = data.file_name;
    audioDurationPill.textContent = formatTime(data.duration_sec);
    sampleRatePill.textContent = `${(data.sample_rate / 1000).toFixed(1)} kHz`;
    totalTimeDisplay.textContent = formatTime(data.duration_sec);
    currentTimeDisplay.textContent = '00:00';

    // Set audio player source
    audio.src = data.audio_url;
    audio.load();

    // Show dashboard
    analysisLoading.style.display = 'none';
    analysisDashboard.style.display = 'block';

    // Render all visual components
    resizeCanvas();
    setupDangerBands();
    renderTranscript();
    renderDangerousSegments();
    renderConclusion();

    // Scroll smoothly to player
    analysisDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Auto-play audio so user experiences real-time sync immediately
    audio.play().then(() => {
      isPlaying = true;
      updatePlayIcons();
      startSyncLoop();
    }).catch(err => {
      console.log('Autoplay requires user interaction:', err);
    });
  }

  // ---------------------------------------------------------------------------
  // 1-Click Demo Handling
  // ---------------------------------------------------------------------------
  sampleCards.forEach(card => {
    card.addEventListener('click', () => {
      const sampleId = card.getAttribute('data-sample-id');
      loadSampleDemo(sampleId);
    });
  });

  async function loadSampleDemo(sampleId) {
    showLoading('Analyzing Demo Call Scenario...', 'Extracting DSP features, checking phase jitter, and running context risk engine...');

    try {
      const response = await fetch(`/api/v1/sample-calls/${sampleId}`);
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      displayAnalysis(data);
    } catch (err) {
      alert(`Failed to load sample call: ${err.message}`);
      analysisLoading.style.display = 'none';
    }
  }

  // ---------------------------------------------------------------------------
  // Audio File Upload Handling (Drag & Drop + File Picker)
  // ---------------------------------------------------------------------------
  dropZone.addEventListener('click', () => audioFileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleAudioUpload(e.dataTransfer.files[0]);
    }
  });

  audioFileInput.addEventListener('change', () => {
    if (audioFileInput.files && audioFileInput.files.length > 0) {
      handleAudioUpload(audioFileInput.files[0]);
    }
  });

  async function handleAudioUpload(file) {
    if (!file) return;

    showLoading(
      `Analyzing "${file.name}"...`,
      'Decoding audio stream, extracting MODGDF phase coherence, CQCC, fast pitch jitter, and screening social engineering keywords...'
    );

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/v1/analyze-upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Upload failed with HTTP ${response.status}`);
      }

      const data = await response.json();
      displayAnalysis(data);
    } catch (err) {
      alert(`Audio Analysis Error: ${err.message}`);
      analysisLoading.style.display = 'none';
    }
  }

  function showLoading(title, desc) {
    if (audio && !audio.paused) {
      audio.pause();
      isPlaying = false;
      updatePlayIcons();
      stopSyncLoop();
    }
    loadingStepTitle.textContent = title;
    loadingStepDesc.textContent = desc;
    analysisLoading.style.display = 'block';
    analysisDashboard.style.display = 'none';
    analysisLoading.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

})();
