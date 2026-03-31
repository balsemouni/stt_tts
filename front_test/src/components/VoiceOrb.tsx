import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Mic, MicOff } from 'lucide-react';

interface VoiceOrbProps {
  isActive: boolean;
  isConnecting: boolean;
  onToggle: () => void;
  rms: number;
}

/* ── Mini brain circuit SVG inside the orb ── */
const OrbBrainSVG = ({ intensity }: { intensity: number }) => (
  <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full pointer-events-none" style={{ opacity: 0.5 + intensity * 0.3 }}>
    <defs>
      <linearGradient id="orbLineC" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.7" />
        <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.2" />
      </linearGradient>
      <linearGradient id="orbLineV" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.7" />
        <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.2" />
      </linearGradient>
    </defs>
    {/* Circuit lines */}
    <line x1="50" y1="28" x2="35" y2="42" stroke="url(#orbLineC)" strokeWidth="0.6" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite' }} />
    <line x1="50" y1="28" x2="65" y2="42" stroke="url(#orbLineV)" strokeWidth="0.6" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.3s' }} />
    <line x1="35" y1="42" x2="50" y2="58" stroke="url(#orbLineV)" strokeWidth="0.6" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.6s' }} />
    <line x1="65" y1="42" x2="50" y2="58" stroke="url(#orbLineC)" strokeWidth="0.6" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.9s' }} />
    <line x1="35" y1="42" x2="25" y2="55" stroke="url(#orbLineC)" strokeWidth="0.4" strokeDasharray="2 4" style={{ animation: 'dataFlow 3s linear infinite' }} />
    <line x1="65" y1="42" x2="75" y2="55" stroke="url(#orbLineV)" strokeWidth="0.4" strokeDasharray="2 4" style={{ animation: 'dataFlow 3s linear infinite 0.5s' }} />
    <line x1="50" y1="58" x2="38" y2="72" stroke="url(#orbLineC)" strokeWidth="0.5" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 1.2s' }} />
    <line x1="50" y1="58" x2="62" y2="72" stroke="url(#orbLineV)" strokeWidth="0.5" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 1.5s' }} />
    {/* Nodes */}
    <circle cx="50" cy="28" r="2.5" fill="#22d3ee" style={{ animation: 'brainPulse 2s ease-in-out infinite' }} />
    <circle cx="35" cy="42" r="2" fill="#a78bfa" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.3s' }} />
    <circle cx="65" cy="42" r="2" fill="#22d3ee" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.6s' }} />
    <circle cx="50" cy="58" r="3" fill="white" opacity="0.9" style={{ animation: 'brainPulse 2.5s ease-in-out infinite' }} />
    <circle cx="25" cy="55" r="1.5" fill="#22d3ee" opacity="0.5" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.8s' }} />
    <circle cx="75" cy="55" r="1.5" fill="#a78bfa" opacity="0.5" style={{ animation: 'brainPulse 2s ease-in-out infinite 1s' }} />
    <circle cx="38" cy="72" r="2" fill="#22d3ee" style={{ animation: 'brainPulse 2s ease-in-out infinite 1.2s' }} />
    <circle cx="62" cy="72" r="2" fill="#a78bfa" style={{ animation: 'brainPulse 2s ease-in-out infinite 1.5s' }} />
    {/* Core glow rings */}
    <circle cx="50" cy="50" r="10" fill="none" stroke="#22d3ee" strokeWidth="0.3" opacity="0.2" style={{ animation: 'brainPulse 3s ease-in-out infinite' }} />
    <circle cx="50" cy="50" r="16" fill="none" stroke="#a78bfa" strokeWidth="0.2" opacity="0.15" style={{ animation: 'brainPulse 4s ease-in-out infinite 0.5s' }} />
  </svg>
);

export const VoiceOrb: React.FC<VoiceOrbProps> = ({ isActive, isConnecting, onToggle, rms }) => {
  const intensity = Math.min(rms, 1);

  return (
    <div className="relative flex items-center justify-center py-6" style={{ perspective: '1000px' }}>
      {/* Deep background glow */}
      <AnimatePresence>
        {isActive && (
          <>
            <motion.div
              initial={{ scale: 0.3, opacity: 0 }}
              animate={{ scale: 1.6 + intensity * 0.6, opacity: 0.12 }}
              exit={{ scale: 0.3, opacity: 0 }}
              transition={{ duration: 0.6 }}
              className="absolute w-80 h-80 rounded-full blur-[100px]"
              style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.35), rgba(139,92,246,0.15), transparent)' }}
            />
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 2 + intensity * 1.5, opacity: 0.05 }}
              exit={{ scale: 0.5, opacity: 0 }}
              transition={{ duration: 1 }}
              className="absolute w-[26rem] h-[26rem] rounded-full blur-[120px]"
              style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.2), transparent)' }}
            />
          </>
        )}
      </AnimatePresence>

      {/* 3D Tilted orbit ring 1 */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1, rotate: 360 }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ rotate: { duration: 5, repeat: Infinity, ease: "linear" }, opacity: { duration: 0.5 } }}
            className="absolute w-48 h-48"
            style={{ transformStyle: 'preserve-3d', transform: 'rotateX(70deg)' }}
          >
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(6,182,212,0.12)" strokeWidth="0.5" />
            </svg>
            {[0, 90, 180, 270].map((deg) => (
              <motion.div key={deg}
                className="absolute rounded-full"
                style={{
                  width: '5px', height: '5px',
                  top: `${50 - 45 * Math.cos(deg * Math.PI / 180)}%`,
                  left: `${50 + 45 * Math.sin(deg * Math.PI / 180)}%`,
                  background: deg % 180 === 0 ? '#22d3ee' : '#a78bfa',
                  boxShadow: `0 0 12px ${deg % 180 === 0 ? 'rgba(34,211,238,0.8)' : 'rgba(167,139,250,0.8)'}`,
                }}
                animate={{ scale: [0.8, 1.5, 0.8] }}
                transition={{ duration: 1.2, repeat: Infinity, delay: deg / 360 }}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3D Tilted orbit ring 2 - counter */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 0.6, scale: 1, rotate: -360 }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ rotate: { duration: 8, repeat: Infinity, ease: "linear" }, opacity: { duration: 0.5 } }}
            className="absolute w-56 h-56"
            style={{ transformStyle: 'preserve-3d', transform: 'rotateX(75deg) rotateZ(60deg)' }}
          >
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(139,92,246,0.08)" strokeWidth="0.5" strokeDasharray="4 8" />
            </svg>
            {[0, 120, 240].map((deg) => (
              <div key={deg}
                className="absolute w-1.5 h-1.5 rounded-full"
                style={{
                  top: `${50 - 45 * Math.cos(deg * Math.PI / 180)}%`,
                  left: `${50 + 45 * Math.sin(deg * Math.PI / 180)}%`,
                  background: '#67e8f9',
                  boxShadow: '0 0 6px rgba(103,232,249,0.5)',
                }}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Orb */}
      <motion.button
        onClick={onToggle}
        disabled={isConnecting}
        whileHover={{ scale: 1.06, rotateX: 8 }}
        whileTap={{ scale: 0.93 }}
        animate={isActive ? { scale: 1 + intensity * 0.04 } : {}}
        className="relative w-36 h-36 md:w-40 md:h-40 rounded-full flex items-center justify-center cursor-pointer"
        style={{
          background: isActive
            ? 'linear-gradient(135deg, #0891b2, #7c3aed, #0891b2)'
            : 'linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))',
          border: isActive ? 'none' : '1px solid rgba(255,255,255,0.08)',
          boxShadow: isActive
            ? `0 0 ${30 + intensity * 35}px rgba(6,182,212,0.25), 0 0 ${60 + intensity * 50}px rgba(139,92,246,0.12), inset 0 0 30px rgba(255,255,255,0.04)`
            : '0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)',
          transformStyle: 'preserve-3d',
          transition: 'box-shadow 0.3s, background 0.5s',
        }}
      >
        {/* Glass highlight */}
        <div className="absolute inset-0 rounded-full overflow-hidden">
          <div className={`absolute inset-0 transition-opacity duration-500 ${isActive ? 'opacity-100' : 'opacity-30'}`}
            style={{ background: 'radial-gradient(ellipse at 30% 20%, rgba(255,255,255,0.15) 0%, transparent 50%)' }} />
        </div>

        {/* Inner dark ring for depth */}
        <div className="absolute inset-[3px] rounded-full"
          style={{
            background: isActive
              ? 'radial-gradient(circle at 40% 35%, rgba(6,182,212,0.15), rgba(2,2,6,0.7) 70%)'
              : 'rgba(2,2,6,0.6)',
            boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.5)',
          }} />

        {/* AI Brain circuit inside active orb */}
        {isActive && <OrbBrainSVG intensity={intensity} />}

        <div className="relative z-10 flex flex-col items-center gap-1.5">
          {isConnecting ? (
            <div className="w-9 h-9 border-2 border-white/15 border-t-neon-400 rounded-full animate-spin" />
          ) : isActive ? (
            <>
              <MicOff size={30} className="text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.5)]" />
              <span className="text-[7px] font-bold text-white/50 uppercase tracking-[0.3em]">Listening...</span>
            </>
          ) : (
            <>
              <Mic size={32} className="text-neon-400 drop-shadow-[0_0_15px_rgba(6,182,212,0.6)]" />
              <span className="text-[7px] font-bold text-dark-300 uppercase tracking-[0.3em]">Tap to speak</span>
            </>
          )}
        </div>

        {/* Pulse rings */}
        {isActive && (
          <>
            <motion.div animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.04, 0.3] }}
              transition={{ duration: 1.6, repeat: Infinity }}
              className="absolute -inset-3 rounded-full" style={{ border: '1px solid rgba(6,182,212,0.2)' }} />
            <motion.div animate={{ scale: [1, 1.3, 1], opacity: [0.2, 0.02, 0.2] }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.3 }}
              className="absolute -inset-6 rounded-full" style={{ border: '1px solid rgba(139,92,246,0.12)' }} />
            <motion.div animate={{ scale: [1, 1.4, 1], opacity: [0.1, 0.01, 0.1] }}
              transition={{ duration: 2.5, repeat: Infinity, delay: 0.6 }}
              className="absolute -inset-9 rounded-full" style={{ border: '1px solid rgba(6,182,212,0.06)' }} />
          </>
        )}
      </motion.button>
    </div>
  );
};
