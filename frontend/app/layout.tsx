import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ClerkProvider } from '@clerk/nextjs';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'HealthGuard XAI — Explainable AI Health Risk Platform',
  description:
    'Research-grade AI for diabetes risk prediction with SHAP explainability, personalized recommendations, and an LLM health assistant.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider afterSignOutUrl="/sign-in">
      <html lang="en" className="h-full">
        <body className={`${inter.className} h-full bg-slate-50 text-slate-900 antialiased`}>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
