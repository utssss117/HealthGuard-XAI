import { UserButton } from '@clerk/nextjs';
import Link from 'next/link';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950">
      {/* Top navbar */}
      <header className="bg-slate-950/70 border-b border-slate-800/60 px-6 py-4 flex items-center justify-between sticky top-0 z-50 backdrop-blur-md">
        <Link href="/dashboard" className="flex items-center gap-3 no-underline group">
          <span className="text-3xl transition-transform group-hover:scale-105 duration-200">🛡️</span>
          <div>
            <span className="font-bold text-white text-lg tracking-tight leading-none block">HealthGuard XAI</span>
            <span className="text-slate-400 text-xs mt-0.5 block">Explainable AI · Health Risk</span>
          </div>
        </Link>
        <div className="flex items-center gap-4">
          <span className="text-xs text-slate-500 font-mono hidden md:block">
            API: {process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}
          </span>
          <UserButton
            appearance={{
              elements: {
                userButtonAvatarBox: "w-9 h-9 border border-slate-800 hover:border-slate-700 transition-colors"
              }
            }}
          />
        </div>
      </header>

      {/* Page content */}
      <main className="flex-1 container mx-auto max-w-6xl px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-slate-900 py-4 text-center text-xs text-slate-500 bg-slate-950/80 backdrop-blur-sm">
        HealthGuard XAI — For research and informational purposes only. Not a substitute for professional medical advice.
      </footer>
    </div>
  );
}
