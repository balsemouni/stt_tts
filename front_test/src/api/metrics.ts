import type { GPUData, LatencyData } from "../types";

// Metrics come from the CAG service (port 8000) or a gateway endpoint
const CAG_URL = import.meta.env.VITE_CAG_URL || "http://localhost:8000";

export const metricsApi = {
  async getGPU(): Promise<GPUData> {
    const res = await fetch(`${CAG_URL}/metrics`);
    if (!res.ok) throw new Error("Metrics unavailable");
    return res.json();
  },

  async getLatency(): Promise<LatencyData> {
    const res = await fetch(`${CAG_URL}/metrics`);
    if (!res.ok) throw new Error("Metrics unavailable");
    return res.json();
  },
};
