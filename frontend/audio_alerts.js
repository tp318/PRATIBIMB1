/**
 * PRATIBIMB Security System Avionics Audio Engine
 *
 * Emits authentic, high-definition electronic security system beeps (Honeywell / ADT / Bosch style)
 * Clean high-frequency piezo acoustics (2400 Hz - 2850 Hz) with razor-sharp attack and clean cutoff.
 * Loud, unmistakable, authentic security alert pulses without harsh distortion.
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
   * Generates a clean, crisp security system piezo beep pulse.
   * @param {AudioContext} ctx
   * @param {number} freq Frequency in Hz (e.g. 2400 - 2850 Hz)
   * @param {number} startTime Scheduled start time in AudioContext time
   * @param {number} duration Duration in seconds (e.g. 0.065s)
   * @param {number} gainVal Volume gain (0.0 to 1.0)
   */
  function scheduleSecurityBeepPulse(ctx, freq, startTime, duration, gainVal) {
    if (gainVal === undefined) gainVal = 0.88;

    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();

    // Security keypad piezo characteristics: sine wave with rapid exponential ramp
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, startTime);

    // Instant attack, sustained punch, fast clean release
    gainNode.gain.setValueAtTime(0.0001, startTime);
    gainNode.gain.linearRampToValueAtTime(gainVal, startTime + 0.003);
    gainNode.gain.setValueAtTime(gainVal, startTime + duration - 0.008);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    // Subtle overtone for authentic metallic piezo keypad resonance
    const oscHarmonic = ctx.createOscillator();
    const gainHarmonic = ctx.createGain();
    oscHarmonic.type = 'sine';
    oscHarmonic.frequency.setValueAtTime(freq * 2, startTime);
    gainHarmonic.gain.setValueAtTime(0.0001, startTime);
    gainHarmonic.gain.linearRampToValueAtTime(gainVal * 0.12, startTime + 0.003);
    gainHarmonic.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    osc.connect(gainNode);
    gainNode.connect(ctx.destination);
    oscHarmonic.connect(gainHarmonic);
    gainHarmonic.connect(ctx.destination);

    osc.start(startTime);
    osc.stop(startTime + duration + 0.01);
    oscHarmonic.start(startTime);
    oscHarmonic.stop(startTime + duration + 0.01);
  }

  /**
   * Dual-chirp security system alert pulse (classic "BEEP-BEEP" security alarm alert).
   */
  function emitSecurityAlarmBeep() {
    if (isMuted) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const now = ctx.currentTime;

      // Pulse 1: 2500 Hz for 65ms
      scheduleSecurityBeepPulse(ctx, 2500, now, 0.065, 0.88);

      // Pulse 2: 2500 Hz for 65ms after 105ms offset
      scheduleSecurityBeepPulse(ctx, 2500, now + 0.105, 0.065, 0.88);
    } catch (e) {
      console.warn('[PratibimbAudio] Security beep error:', e);
    }
  }

  /**
   * Single or double confirmation security keypad chime.
   */
  function emitKeypadChime() {
    if (isMuted) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const now = ctx.currentTime;

      // Keypad armed / confirmation: 2400 Hz then 2850 Hz ascending chirp
      scheduleSecurityBeepPulse(ctx, 2400, now, 0.055, 0.75);
      scheduleSecurityBeepPulse(ctx, 2850, now + 0.075, 0.070, 0.80);
    } catch (e) {
      console.warn('[PratibimbAudio] Keypad chime error:', e);
    }
  }

  const PratibimbAudio = {
    /**
     * Starts continuous security system warning beep while active.
     */
    startContinuousBeep: function (intervalMs) {
      if (intervalMs === undefined) intervalMs = 600;
      if (beepIntervalId) return; // already active
      emitSecurityAlarmBeep();
      beepIntervalId = setInterval(function () {
        emitSecurityAlarmBeep();
      }, intervalMs);
    },

    /**
     * Immediately stops the security alert sequence.
     */
    stopContinuousBeep: function () {
      if (beepIntervalId) {
        clearInterval(beepIntervalId);
        beepIntervalId = null;
      }
    },

    /**
     * Checks if the continuous alert sequence is currently active.
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
     * Play a single confirmation security keypad chirp.
     */
    playChime: function () {
      emitKeypadChime();
    }
  };

  window.PratibimbAudio = PratibimbAudio;
})(window);
