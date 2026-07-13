export default function ApiError({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-4 py-3 mb-4">
      <span className="shrink-0 mt-0.5">⚠️</span>
      <div>
        <p className="font-medium">Request failed</p>
        <p className="text-xs text-red-500 mt-0.5 font-mono break-all">{message}</p>
      </div>
    </div>
  );
}
