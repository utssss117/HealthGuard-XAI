import { auth, currentUser } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import DashboardClient from "./components/DashboardClient";

export default async function DashboardPage() {
  const { userId, getToken } = await auth();

  // If the user isn't logged in, redirect them to sign-in
  if (!userId) {
    redirect("/sign-in");
  }

  // Get user details
  const user = await currentUser();
  const token = await getToken();

  // Pass token to client component for client-side sync
  return (
    <main className="min-h-screen bg-[#020617] text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-12 gap-4">
          <div>
            <h1 className="text-3xl font-extrabold text-[#f8fafc] tracking-tight">Patient Dashboard</h1>
            <p className="text-[#94a3b8] mt-1 text-sm font-medium">
              Welcome back, {user?.firstName || user?.emailAddresses?.[0]?.emailAddress || "User"}
            </p>
          </div>
        </div>
 
        {/* Client Application */}
        <DashboardClient authToken={token || ""} />
      </div>
    </main>
  );
}
