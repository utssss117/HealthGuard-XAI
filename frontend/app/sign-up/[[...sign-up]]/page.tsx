import { SignUp } from '@clerk/nextjs';

export default function SignUpPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 px-4">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <span className="text-3xl">🛡️</span>
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight">HealthGuard XAI</h1>
        </div>
        <p className="text-slate-500 text-sm">Create your account to get started</p>
      </div>
      <SignUp fallbackRedirectUrl="/dashboard" />
    </div>
  );
}
