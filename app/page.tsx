import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { ArrowRight, Shield, Zap, BarChart3, HeartPulse } from "lucide-react";
import Link from "next/link";
import { SignInButton, SignUpButton } from "@clerk/nextjs";

export default async function HomePage() {
  const { userId } = await auth();

  if (userId) {
    redirect("/dashboard");
  }

  return (
    <div className="flex flex-col min-h-screen bg-[#020617] overflow-hidden">
      {/* Hero Section */}
      <section className="relative pt-20 pb-20 md:pt-32 md:pb-32 px-6 overflow-hidden">
        {/* Animated Background Orbs */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[600px] pointer-events-none opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#6c47ff] rounded-full blur-[120px] animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-[#8b5cf6] rounded-full blur-[100px] animate-pulse [animation-delay:2s]"></div>
        </div>

        <div className="max-w-7xl mx-auto relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs font-bold text-[#6c47ff] mb-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <span className="flex h-2 w-2 rounded-full bg-[#6c47ff] animate-ping"></span>
            Next Generation AI Diagnostics
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-white mb-6 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-100">
            Precision Healthcare <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-[#6c47ff] via-[#8b5cf6] to-[#a78bfa]">
              Powered by Explainable AI
            </span>
          </h1>

          <p className="max-w-2xl mx-auto text-lg md:text-xl text-[#94a3b8] mb-12 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-200">
            Predict patient risks with world-class accuracy and understand the "why" behind every diagnosis with our advanced medical XAI engine.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300">
            <SignUpButton mode="modal">
              <button className="w-full sm:w-auto px-8 py-4 bg-[#6c47ff] hover:bg-[#5b3ae6] text-white rounded-full font-bold text-lg shadow-xl shadow-[#6c47ff]/20 flex items-center justify-center gap-2 transition-all transform hover:scale-105 active:scale-95">
                Join HealthGuard <ArrowRight size={20} />
              </button>
            </SignUpButton>
            <Link href="#features" className="w-full sm:w-auto px-8 py-4 bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-full font-bold text-lg transition-all">
              Explore Features
            </Link>
          </div>
        </div>

        {/* Mock Dashboard Preview */}
        <div className="max-w-5xl mx-auto mt-20 relative animate-in fade-in slide-in-from-bottom-12 duration-1000 delay-500">
           <div className="absolute inset-0 bg-gradient-to-t from-[#020617] via-transparent to-transparent z-10 h-full w-full"></div>
           <div className="p-1 rounded-3xl bg-gradient-to-b from-white/20 to-transparent shadow-2xl overflow-hidden">
             <div className="bg-[#0f172a] rounded-[22px] min-h-[400px] flex items-center justify-center border border-white/5 relative overflow-hidden">
               {/* Background patterns */}
               <div className="absolute inset-0 opacity-[0.03]" style={{backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '24px 24px'}}></div>
               <div className="text-center p-12 relative z-20">
                 <div className="w-16 h-16 bg-[#6c47ff]/20 rounded-2xl flex items-center justify-center text-[#6c47ff] mx-auto mb-6 border border-[#6c47ff]/30 backdrop-blur-sm">
                   <BarChart3 size={32} />
                 </div>
                 <h3 className="text-2xl font-bold text-white mb-2 font-mono tracking-wider">SECURE_DASHBOARD_LIVE</h3>
                 <p className="text-[#64748b] font-medium uppercase tracking-[0.2em] text-xs">Awaiting Clinical Authentication</p>
               </div>
             </div>
           </div>
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" className="py-24 px-6 bg-black/40 border-t border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { icon: HeartPulse, title: "Risk Prediction", desc: "Advanced neural networks trained on clinical datasets for high-accuracy patient assessments." },
              { icon: Shield, title: "XAI Transparency", desc: "Understand exactly which biomarkers are driving patient risk with integrated feature importance hooks." },
              { icon: Zap, title: "Real-time AI", desc: "Get instant diagnostic insights and tailored lifestyle recommendations in milliseconds." },
              { icon: BarChart3, title: "Clinical Insights", desc: "Visualize patient health telemetry with normalized radar charts and biomarker tracking." },
            ].map((feature, i) => (
              <div key={i} className="group p-8 rounded-3xl bg-white/5 border border-white/5 hover:border-[#6c47ff]/30 transition-all duration-300">
                <div className="w-12 h-12 rounded-2xl bg-[#1e293b] border border-white/10 flex items-center justify-center text-[#6c47ff] mb-6 group-hover:bg-[#6c47ff] group-hover:text-white transition-all">
                  <feature.icon size={24} />
                </div>
                <h3 className="text-xl font-bold text-white mb-3 tracking-tight">{feature.title}</h3>
                <p className="text-[#94a3b8] text-sm leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5 text-center">
        <div className="flex items-center justify-center gap-3 mb-6 opacity-60 grayscale hover:grayscale-0 transition-all cursor-pointer">
          <div className="w-8 h-8 bg-[#6c47ff] rounded-lg"></div>
          <span className="font-bold text-white">HealthGuard XAI</span>
        </div>
        <p className="text-[#475569] text-sm">© 2026 HealthGuard Intelligence Systems. All rights reserved.</p>
      </footer>
    </div>
  );
}
