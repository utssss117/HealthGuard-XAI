import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-80px)] bg-[#020617] px-6 relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-[500px] pointer-events-none opacity-10">
        <div className="absolute top-0 left-0 w-80 h-80 bg-[#6c47ff] rounded-full blur-[100px]"></div>
        <div className="absolute bottom-0 right-0 w-80 h-80 bg-[#8b5cf6] rounded-full blur-[100px]"></div>
      </div>

      <div className="relative z-10 w-full max-w-md animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="text-center mb-10">
          <div className="w-16 h-16 bg-gradient-to-br from-[#6c47ff] to-[#8b5cf6] rounded-2xl flex items-center justify-center text-white shadow-xl shadow-[#6c47ff]/20 mx-auto mb-6">
            <span className="text-3xl font-bold">⚕</span>
          </div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Create Account</h1>
          <p className="text-[#94a3b8] mt-2">Join the future of diagnostic intelligence</p>
        </div>

        <SignUp 
          appearance={{
            elements: {
              rootBox: "w-full",
              card: "shadow-none border-none bg-transparent p-0",
              header: "hidden", 
              footer: "bg-transparent",
              formButtonPrimary: "w-full h-12 rounded-xl text-base font-bold tracking-tight",
              formFieldInput: "h-12 rounded-xl border-[#1e293b] bg-[#0f172a] text-white focus:border-[#6c47ff] placeholder:text-[#475569]",
              socialButtonsBlockButton: "h-12 rounded-xl border-[#1e293b] bg-[#1e293b] text-white hover:bg-[#334155]",
              socialButtonsBlockButtonText: "text-white font-bold !opacity-100",
              dividerLine: "bg-[#1e293b]",
              dividerText: "text-[#94a3b8] bg-[#020617] font-bold !opacity-100",
              footerActionText: "text-[#94a3b8] !opacity-100",
              footerActionLink: "text-[#6c47ff] hover:text-[#5b3ae6] font-bold",
              formFieldLabel: "text-white font-bold !opacity-100 mb-2",
              formFieldInputShowPasswordButton: "text-[#94a3b8] hover:text-white",
            }
          }}
          signInUrl="/sign-in"
          forceRedirectUrl="/dashboard"
        />
      </div>
    </div>
  );
}
