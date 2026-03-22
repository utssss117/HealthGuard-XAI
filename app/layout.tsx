import type { Metadata } from 'next'
import { ClerkProvider, SignInButton, SignUpButton, UserButton } from '@clerk/nextjs'
import { auth } from '@clerk/nextjs/server'
import { Geist, Geist_Mono } from 'next/font/google'
import Link from 'next/link'
import './globals.css'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'HealthGuard-XAI',
  description: 'AI-driven Health Dashboard',
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  const { userId } = await auth();

  return (
    <ClerkProvider
      appearance={{
        baseTheme: undefined, // Custom dark theme below
        variables: {
          colorPrimary: '#6c47ff',
          colorText: '#ffffff',
          colorBackground: '#020617',
          colorInputBackground: '#0f172a',
          colorInputText: '#ffffff',
          colorTextOnPrimaryBackground: '#ffffff',
          colorTextSecondary: '#94a3b8',
        },
        elements: {
          card: 'bg-[#0f172a] border border-[#1e293b] shadow-2xl rounded-3xl',
          headerTitle: 'text-2xl font-bold tracking-tight text-white !opacity-100',
          headerSubtitle: 'text-[#94a3b8] !opacity-100',
          socialButtonsBlockButton: 'bg-[#1e293b] border-[#334155] hover:bg-[#334155] text-white',
          socialButtonsBlockButtonText: 'text-white font-semibold !opacity-100',
          formButtonPrimary: 'bg-[#6c47ff] hover:bg-[#5b3ae6] transition-all duration-300 text-white font-bold',
          footerActionLink: 'text-[#6c47ff] hover:text-[#5b3ae6] font-semibold',
          footerActionText: 'text-[#94a3b8] !opacity-100',
          dividerLine: 'bg-[#1e293b]',
          dividerText: 'text-[#64748b] bg-[#020617] !opacity-100',
          formFieldLabel: 'text-white font-bold opacity-100 block mb-2 transition-all',
          formFieldInput: 'bg-[#0f172a] border-[#1e293b] focus:border-[#6c47ff] text-white placeholder:text-[#475569]',
          identityPreviewText: 'text-white !opacity-100',
          identityPreviewEditButtonIcon: 'text-[#6c47ff]',
          formFieldSuccessText: 'text-emerald-400',
          formFieldErrorText: 'text-rose-400',
          breadcrumbsItem: 'text-[#94a3b8]',
          breadcrumbsSeparator: 'text-[#475569]',
        }
      }}
    >
      <html lang="en">
        <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#020617] text-white selection:bg-[#6c47ff]/30`}>
          <header className={`fixed top-0 left-0 right-0 z-50 flex justify-between items-center px-6 sm:px-12 h-20 transition-all duration-300 border-b border-white/5 bg-[#020617]/80 backdrop-blur-xl`}>
            <Link href="/" className="flex items-center gap-3 group cursor-pointer">
              <div className="w-10 h-10 bg-gradient-to-br from-[#6c47ff] to-[#8b5cf6] rounded-xl flex items-center justify-center text-white shadow-lg shadow-[#6c47ff]/20 group-hover:scale-110 transition-transform duration-300">
                <span className="text-xl font-bold">⚕</span>
              </div>
              <span className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-[#94a3b8]">
                HealthGuard <span className="text-[#6c47ff]">XAI</span>
              </span>
            </Link>

            <div className="flex items-center gap-6">
              {!userId ? (
                <div className="flex items-center gap-4">
                  <Link href="/sign-in" className="text-sm font-semibold text-[#94a3b8] hover:text-white transition-colors">
                    Sign In
                  </Link>
                  <Link href="/sign-up">
                    <button className="bg-[#6c47ff] hover:bg-[#5b3ae6] text-white text-sm font-bold px-6 py-2.5 rounded-full shadow-lg shadow-[#6c47ff]/20 transition-all transform hover:-translate-y-0.5 active:translate-y-0">
                      Get Started
                    </button>
                  </Link>
                </div>
              ) : (
                <div className="flex items-center gap-6">
                  <nav className="hidden md:flex items-center gap-6">
                    <a href="/dashboard" className="text-sm font-semibold text-[#6c47ff] hover:text-purple-400 transition-colors">Dashboard</a>
                  </nav>
                  <UserButton 
                    appearance={{ 
                      elements: { 
                        userButtonAvatarBox: "w-10 h-10 border-2 border-[#1e293b] hover:border-[#6c47ff] transition-all",
                        userButtonTrigger: "focus:shadow-none"
                      } 
                    }} 
                  />
                </div>
              )}
            </div>
          </header>
          
          <main className="pt-20">
            {children}
          </main>
        </body>
      </html>
    </ClerkProvider>
  )
}

