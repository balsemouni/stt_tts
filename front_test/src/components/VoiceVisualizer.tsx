import React from "react";
import { motion } from "motion/react";

interface VoiceVisualizerProps {
  isActive: boolean;
  volume?: number;
}

export const VoiceVisualizer: React.FC<VoiceVisualizerProps> = ({ isActive, volume = 0 }) => {
  const bars = Array.from({ length: 12 });

  return (
    <div className="wave-container">
      {bars.map((_, i) => (
        <motion.div
          key={i}
          className="wave-bar"
          animate={{
            height: isActive ? [10, 20 + Math.random() * 30, 10] : 4,
            opacity: isActive ? 1 : 0.2,
          }}
          transition={{
            duration: 0.5,
            repeat: Infinity,
            delay: i * 0.05,
            ease: "easeInOut",
          }}
          style={{
            background: `linear-gradient(to top, rgba(6,182,212,${isActive ? 0.8 : 0.3}), rgba(139,92,246,${isActive ? 0.8 : 0.3}))`,
            boxShadow: isActive ? '0 0 6px rgba(6,182,212,0.3)' : 'none',
          }}
        />
      ))}
    </div>
  );
};
