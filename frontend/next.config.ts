import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    // Anchor Turbopack to this subdirectory so it doesn't pick up
    // the parent repo's package-lock.json as the workspace root
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
