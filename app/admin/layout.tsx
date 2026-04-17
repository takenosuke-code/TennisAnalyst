// Admin API routes are password-guarded per request via requireAdminAuth().
// The layout no longer blocks rendering — the page itself handles the
// password prompt and locks the UI until a valid token is stored. Anyone
// loading the page without the password sees only the prompt, not any
// admin controls.
export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}
