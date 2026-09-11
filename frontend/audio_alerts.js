/**
 * PRATIBIMB Avionics Audio Alerts Library
 * 
 * Provides clean, periodic, non-blocking audio alerts for engine health monitoring.
 * Zero external dependencies: uses native Web Audio API with automatic fallback.
 */

(function (window) {
  'use strict';

  let audioCtx = null;
  let beepIntervalId = null;
  let isMuted = false;

  function getAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume().catch(function () {});
    }
    return audioCtx;
  }

  /**
   * Emits a single, clean avionics warning beep tone (960 Hz sine wave, 110ms with smooth envelope)
   */
  function emitBeepTone(freq, durationMs) {
    if (freq === undefined) freq = 960;
    if (durationMs === undefined) durationMs = 110;
    if (isMuted) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;

      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now);

      // Smooth attack and decay to prevent audio clicks
      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.18, now + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.001, now + (durationMs / 1000));

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + (durationMs / 1000) + 0.01);
    } catch (e) {
      console.warn('[PratibimbAudio] Beep error:', e);
    }
  }

  const PratibimbAudio = {
    /**
     * Starts a continuous periodic beep sequence that sounds every intervalMs as long as fault is active.
     */
    startContinuousBeep: function (intervalMs) {
      if (intervalMs === undefined) intervalMs = 700;
      if (beepIntervalId) return; // already beeping
      // Emit immediate first beep
      emitBeepTone(960, 110);
      beepIntervalId = setInterval(function () {
        emitBeepTone(960, 110);
      }, intervalMs);
    },

    /**
     * Immediately stops the continuous beep sequence.
     */
    stopContinuousBeep: function () {
      if (beepIntervalId) {
        clearInterval(beepIntervalId);
        beepIntervalId = null;
      }
    },

    /**
     * Checks if the continuous beep sequence is currently active.
     */
    isBeeping: function () {
      return beepIntervalId !== null;
    },

    /**
     * Mute or unmute all audio alerts.
     */
    setMuted: function (muted) {
      isMuted = Boolean(muted);
    },

    /**
     * Toggle mute state.
     */
    toggleMute: function () {
      isMuted = !isMuted;
      return isMuted;
    },

    getMuted: function () {
      return isMuted;
    },

    /**
     * Play a single confirmation/acknowledgement chime.
     */
    playChime: function () {
      emitBeepTone(1200, 80);
    }
  };

  window.PratibimbAudio = PratibimbAudio;
})(window);
