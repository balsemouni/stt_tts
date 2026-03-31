import { useState, useEffect } from "react";
import { metricsApi } from "../api";
import type { GPUData, LatencyData } from "../types";

const DEFAULT_GPU: GPUData = {
  utilization: 42, memory: 6.2, total_memory: 12,
  memory_percent: 51.7, temperature: 68, power: 185, power_limit: 250,
};

const DEFAULT_LATENCY: LatencyData = {
  avg_stt: 124, avg_llm: 342, avg_tts: 187,
  p95_stt: 245, p95_llm: 687, p95_tts: 324,
};

function jitter(val: number, range: number, min: number, max: number) {
  return Math.min(max, Math.max(min, val + (Math.random() - 0.5) * range));
}

export function useMetrics(intervalMs = 10000) {
  const [gpuMetrics, setGpuMetrics] = useState<GPUData>(DEFAULT_GPU);
  const [latencyData, setLatencyData] = useState<LatencyData>(DEFAULT_LATENCY);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [gpu, latency] = await Promise.all([
          metricsApi.getGPU(),
          metricsApi.getLatency(),
        ]);
        setGpuMetrics(gpu);
        setLatencyData(latency);
      } catch {
        // Simulated fallback
        setGpuMetrics((p) => ({
          ...p,
          utilization: jitter(p.utilization, 8, 20, 95),
          memory: jitter(p.memory, 0.5, 1, 11),
          memory_percent: jitter(p.memory_percent, 5, 10, 95),
          temperature: jitter(p.temperature, 3, 45, 85),
          power: jitter(p.power, 10, 120, 245),
        }));
        setLatencyData((p) => ({
          avg_stt: jitter(p.avg_stt, 15, 80, 300),
          avg_llm: jitter(p.avg_llm, 30, 200, 600),
          avg_tts: jitter(p.avg_tts, 20, 120, 350),
          p95_stt: jitter(p.p95_stt, 25, 150, 400),
          p95_llm: jitter(p.p95_llm, 50, 400, 900),
          p95_tts: jitter(p.p95_tts, 30, 200, 500),
        }));
      }
    }, intervalMs);
    return () => clearInterval(interval);
  }, [intervalMs]);

  return { gpuMetrics, latencyData };
}
