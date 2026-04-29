<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Modal deploy runbook

The pose-extraction endpoint at `railway-service/modal_inference.py` ships as a Modal class (`@app.cls` + `@modal.fastapi_endpoint`). Modal generates a different public URL for class-method endpoints than for plain function endpoints, AND any rename of the class or method changes the URL again.

After every `modal deploy railway-service/modal_inference.py`:

1. Read the new endpoint URL from the deploy output. Current shape:
   `https://<workspace>--tennis-pose-inference-poseinference-extract-pose.modal.run`
2. Update Railway's `MODAL_INFERENCE_URL` env var to that exact URL.
3. Redeploy Railway (or wait for it to pick up the env change on next restart).

Skipping step 2 silently regresses every upload to inline Railway CPU extraction — no error surfaces to the user, but Modal GPU never runs and latency is back to 30–90s. The fallback is safe; the silent regression is the risk.