"use client";

import { motion } from "framer-motion";
import { Code2, Copy, Check, Terminal, Key, Zap, BookOpen, ArrowRight, Shield, Activity, CheckCircle2, XCircle, Loader2, Play, Upload } from "lucide-react";
import Link from "next/link";
import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const ML_BASE = "http://localhost:8000";

// ─── Reusable Components ───
function CopyButton({ text }: { text: string }) {
    const [copied, setCopied] = useState(false);
    const handleCopy = () => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); };
    return (
        <button onClick={handleCopy} className="absolute top-3 right-3 p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/80 transition-all" aria-label="Copy code">
            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
        </button>
    );
}

function CodeBlock({ code, language = "bash" }: { code: string; language?: string }) {
    return (
        <div className="relative rounded-xl bg-[#0d1117] border border-white/10 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5 bg-white/[0.02]">
                <Terminal className="w-3.5 h-3.5 text-white/30" /><span className="text-xs font-mono text-white/30">{language}</span>
            </div>
            <pre className="p-4 overflow-x-auto text-sm font-mono text-white/80 leading-relaxed"><code>{code}</code></pre>
            <CopyButton text={code} />
        </div>
    );
}

function StatusDot({ ok }: { ok: boolean | null }) {
    if (ok === null) return <span className="w-2 h-2 rounded-full bg-muted-foreground/30 animate-pulse" />;
    return ok ? <span className="w-2 h-2 rounded-full bg-green-500" /> : <span className="w-2 h-2 rounded-full bg-destructive" />;
}

// ─── Live Health Check Hook ───
function useServiceHealth(url: string) {
    const [status, setStatus] = useState<{ ok: boolean | null; data: Record<string, unknown> | null; latency: number | null; error: string | null }>({ ok: null, data: null, latency: null, error: null });

    const check = useCallback(async () => {
        setStatus(prev => ({ ...prev, ok: null, error: null }));
        const start = performance.now();
        try {
            const res = await axios.get(url, { timeout: 5000 });
            setStatus({ ok: true, data: res.data, latency: Math.round(performance.now() - start), error: null });
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : "Connection failed";
            setStatus({ ok: false, data: null, latency: null, error: msg });
        }
    }, [url]);

    useEffect(() => { check(); }, [check]);
    return { ...status, refresh: check };
}

// ─── Live API Tester ───
function ApiTester() {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [activeEndpoint, setActiveEndpoint] = useState<"analyze" | "history" | "health">("health");

    const runTest = async () => {
        setLoading(true); setResponse(null); setError(null);
        try {
            let res;
            if (activeEndpoint === "health") {
                res = await axios.get(ML_BASE, { timeout: 5000 });
            } else if (activeEndpoint === "history") {
                res = await axios.get(`${API_BASE}/detection/history`, { timeout: 5000 });
            } else if (activeEndpoint === "analyze" && file) {
                const fd = new FormData(); fd.append("file", file);
                res = await axios.post(`${API_BASE}/detection/analyze`, fd, { headers: { "Content-Type": "multipart/form-data" }, timeout: 30000 });
            } else {
                setError("Select a file first"); setLoading(false); return;
            }
            setResponse(JSON.stringify(res.data, null, 2));
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Request failed");
        } finally { setLoading(false); }
    };

    return (
        <div className="card-elevated rounded-xl overflow-hidden">
            <div className="px-5 py-4 flex items-center justify-between" style={{ borderBottom: "1px solid hsl(var(--border) / 0.4)" }}>
                <h3 className="text-heading font-semibold text-foreground flex items-center gap-2"><Play className="w-4 h-4 text-accent" /> Live API Tester</h3>
            </div>
            <div className="p-5 space-y-4">
                <div className="flex gap-2 flex-wrap">
                    {(["health", "history", "analyze"] as const).map(ep => (
                        <button key={ep} onClick={() => setActiveEndpoint(ep)}
                            className={`px-3.5 py-1.5 rounded-md text-[13px] font-medium transition-all ${activeEndpoint === ep ? "bg-foreground text-background" : "bg-muted/60 text-muted-foreground hover:bg-muted"}`}>
                            {ep === "health" ? "GET / (ML Health)" : ep === "history" ? "GET /detection/history" : "POST /detection/analyze"}
                        </button>
                    ))}
                </div>

                {activeEndpoint === "analyze" && (
                    <div className="flex items-center gap-3">
                        <label className="flex items-center gap-2 px-4 py-2 rounded-lg border border-dashed border-border cursor-pointer hover:bg-muted/30 transition-colors text-[13px] text-muted-foreground">
                            <Upload className="w-4 h-4" />
                            {file ? file.name : "Choose image..."}
                            <input type="file" accept="image/*" className="hidden" onChange={e => setFile(e.target.files?.[0] || null)} />
                        </label>
                    </div>
                )}

                <button onClick={runTest} disabled={loading || (activeEndpoint === "analyze" && !file)}
                    className="btn-primary text-[13px] px-5 py-2 disabled:opacity-40 disabled:cursor-not-allowed">
                    {loading ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running...</> : <><Play className="w-3.5 h-3.5" /> Send Request</>}
                </button>

                {(response || error) && (
                    <div className="relative rounded-xl bg-[#0d1117] border border-white/10 overflow-hidden max-h-[400px]">
                        <div className="flex items-center gap-2 px-4 py-2 border-b border-white/5 bg-white/[0.02]">
                            {error ? <XCircle className="w-3.5 h-3.5 text-red-400" /> : <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />}
                            <span className="text-xs font-mono text-white/40">{error ? "Error" : "Response (live)"}</span>
                        </div>
                        <pre className="p-4 overflow-auto text-sm font-mono leading-relaxed max-h-[340px]">
                            <code className={error ? "text-red-400" : "text-white/80"}>{error || response}</code>
                        </pre>
                        {response && <CopyButton text={response} />}
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── Endpoint definitions (from actual backend controllers) ───
const endpoints = [
    {
        method: "POST", path: "/detection/analyze", source: "DetectionController.java",
        description: "Analyze a single image for deepfake detection. Accepts multipart/form-data with a file field. Optionally pass userId to associate the scan with a user.",
        params: [{ name: "file", type: "MultipartFile", required: true, desc: "Image to analyze (JPEG/PNG/WEBP)" }, { name: "userId", type: "Long", required: false, desc: "User ID to associate with scan" }],
        curlExample: `curl -X POST ${API_BASE}/detection/analyze \\\n  -F "file=@image.jpg" \\\n  -F "userId=1"`,
        responseExample: `// Live response — use the API Tester above to see real output`,
    },
    {
        method: "GET", path: "/detection/history", source: "DetectionController.java",
        description: "Retrieve all scan history entries. Returns an array of DetectionHistory objects with scan ID, result label, confidence, metadata, and timestamp.",
        params: [],
        curlExample: `curl ${API_BASE}/detection/history`,
        responseExample: `// Returns: DetectionHistory[] — try it in the API Tester`,
    },
    {
        method: "GET", path: "/detection/history/{scanId}", source: "DetectionController.java",
        description: "Retrieve a specific scan result by its scan ID for detailed report viewing.",
        params: [{ name: "scanId", type: "Long", required: true, desc: "Scan ID from history" }],
        curlExample: `curl ${API_BASE}/detection/history/42`,
        responseExample: `// Returns: DetectionHistory object`,
    },
    {
        method: "POST", path: "/api/users/register", source: "UserController.java",
        description: "Register a new user account. Accepts a JSON body with user details.",
        params: [{ name: "name", type: "String", required: true, desc: "Username" }, { name: "email", type: "String", required: true, desc: "Email address" }, { name: "passwordHash", type: "String", required: false, desc: "Password hash" }],
        curlExample: `curl -X POST ${API_BASE}/api/users/register \\\n  -H "Content-Type: application/json" \\\n  -d '{"name":"demo","email":"demo@test.com"}'`,
        responseExample: `// Returns: Users object`,
    },
    {
        method: "GET", path: "/api/users/{name}", source: "UserController.java",
        description: "Retrieve user details by their username.",
        params: [{ name: "name", type: "String", required: true, desc: "Username to look up" }],
        curlExample: `curl ${API_BASE}/api/users/demo`,
        responseExample: `// Returns: UserDto object`,
    },
    {
        method: "POST", path: "/api/users/clerk-sync", source: "UserController.java",
        description: "Create or update a user from Clerk authentication data. Used for SSO sync.",
        params: [{ name: "clerkId", type: "String", required: true, desc: "Clerk user ID" }, { name: "name", type: "String", required: true, desc: "Display name" }, { name: "email", type: "String", required: true, desc: "Email" }],
        curlExample: `curl -X POST ${API_BASE}/api/users/clerk-sync \\\n  -H "Content-Type: application/json" \\\n  -d '{"clerkId":"clerk_xxx","name":"demo","email":"demo@test.com"}'`,
        responseExample: `// Returns: UserDto object`,
    },
    {
        method: "GET", path: "/ (ML Health)", source: "ml_service/main.py",
        description: "Health check for the ML microservice. Returns model status, device info, and whether the custom or fallback model is loaded.",
        params: [],
        curlExample: `curl ${ML_BASE}/`,
        responseExample: `// Returns: { status, service, model, using_custom_model, device }`,
    },
    {
        method: "POST", path: "/detect (ML Direct)", source: "ml_service/main.py",
        description: "Direct inference endpoint on the ML service. Used internally by the Java backend. Accepts an image file and optional include_ela_image flag.",
        params: [{ name: "file", type: "UploadFile", required: true, desc: "Image file" }, { name: "include_ela_image", type: "bool", required: false, desc: "Include base64 ELA image" }],
        curlExample: `curl -X POST ${ML_BASE}/detect \\\n  -F "file=@image.jpg" \\\n  -F "include_ela_image=true"`,
        responseExample: `// Returns: { filename, verdict, confidence, raw_output, ela, processing_time_ms }`,
    },
];

const quickstartCode: Record<string, string> = {
    rest: `# Analyze an image\ncurl -X POST ${API_BASE}/detection/analyze \\\n  -F "file=@suspicious_photo.jpg"\n\n# Check scan history\ncurl ${API_BASE}/detection/history\n\n# ML service health\ncurl ${ML_BASE}/`,
    python: `import requests\n\n# Analyze an image\nurl = "${API_BASE}/detection/analyze"\nfiles = {"file": open("suspicious_photo.jpg", "rb")}\nresponse = requests.post(url, files=files)\nresult = response.json()\n\nprint(f"Prediction: {result['prediction']}")\nprint(f"Confidence: {result['confidence']}%")\n\nif result["is_deepfake"]:\n    print("⚠️  AI-generated!")\nelse:\n    print("✅  Authentic.")`,
    js: `const formData = new FormData();\nformData.append("file", fileInput.files[0]);\n\nconst response = await fetch(\n  "${API_BASE}/detection/analyze",\n  { method: "POST", body: formData }\n);\n\nconst result = await response.json();\nconsole.log(\`Prediction: \${result.prediction}\`);\nconsole.log(\`Confidence: \${result.confidence}%\`);`,
};

// ─── Main Page ───
export default function DocsPage() {
    const [activeTab, setActiveTab] = useState<"rest" | "python" | "js">("rest");
    const backend = useServiceHealth(API_BASE + "/detection/history");
    const ml = useServiceHealth(ML_BASE);
    const [scanCount, setScanCount] = useState<number | null>(null);

    useEffect(() => {
        if (backend.ok && backend.data && Array.isArray(backend.data)) {
            setScanCount(backend.data.length);
        }
    }, [backend.ok, backend.data]);

    return (
        <div className="flex flex-col bg-background min-h-screen relative overflow-hidden">
            <div className="ambient-orb ambient-orb-cool w-[400px] h-[400px] -top-[100px] -right-[100px] absolute" />

            {/* Hero */}
            <section className="w-full px-6 pt-20 pb-14 relative z-10">
                <div className="max-w-[860px] mx-auto">
                    <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }} className="space-y-6">
                        <span className="section-pill"><Code2 className="w-3.5 h-3.5 text-accent" /> Developer Documentation</span>
                        <h1 className="text-display-lg font-serif text-foreground">API Reference</h1>
                        <p className="text-body-lg text-muted-foreground max-w-[560px] text-balance">
                            Live documentation sourced from your running backend controllers. All endpoints, parameters, and responses pulled from the actual codebase.
                        </p>

                        {/* Live service status */}
                        <div className="flex flex-wrap gap-3 pt-2">
                            <button onClick={backend.refresh} className="flex items-center gap-2 text-[13px] text-muted-foreground px-3.5 py-2 rounded-lg border hover:bg-muted/30 transition-colors" style={{ background: 'hsl(var(--muted) / 0.4)', borderColor: 'hsl(var(--border) / 0.5)' }}>
                                <StatusDot ok={backend.ok} />
                                <Shield className="w-3.5 h-3.5 text-green-500" />
                                Backend <code className="font-mono text-foreground text-[12px]">:8080</code>
                                {backend.latency !== null && <span className="text-[11px] text-green-500">{backend.latency}ms</span>}
                                {backend.ok === false && <span className="text-[11px] text-destructive">offline</span>}
                            </button>
                            <button onClick={ml.refresh} className="flex items-center gap-2 text-[13px] text-muted-foreground px-3.5 py-2 rounded-lg border hover:bg-muted/30 transition-colors" style={{ background: 'hsl(var(--muted) / 0.4)', borderColor: 'hsl(var(--border) / 0.5)' }}>
                                <StatusDot ok={ml.ok} />
                                <Zap className="w-3.5 h-3.5 text-orange-500" />
                                ML Service <code className="font-mono text-foreground text-[12px]">:8000</code>
                                {ml.latency !== null && <span className="text-[11px] text-green-500">{ml.latency}ms</span>}
                                {ml.ok === false && <span className="text-[11px] text-destructive">offline</span>}
                            </button>
                            {scanCount !== null && (
                                <div className="flex items-center gap-2 text-[13px] text-muted-foreground px-3.5 py-2 rounded-lg border" style={{ background: 'hsl(var(--muted) / 0.4)', borderColor: 'hsl(var(--border) / 0.5)' }}>
                                    <Activity className="w-3.5 h-3.5 text-accent" />
                                    <span className="font-mono text-foreground font-semibold text-[12px]">{scanCount}</span> scans in DB
                                </div>
                            )}
                        </div>

                        {/* Live ML model info */}
                        {ml.ok && ml.data && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-lg p-3 text-[12px] font-mono space-y-1" style={{ background: 'hsl(var(--muted) / 0.3)', border: '1px solid hsl(var(--border) / 0.3)' }}>
                                <div className="text-muted-foreground">Model: <span className="text-foreground">{String(ml.data.model)}</span></div>
                                <div className="text-muted-foreground">Device: <span className="text-foreground">{String(ml.data.device)}</span> • Custom: <span className="text-foreground">{String(ml.data.using_custom_model)}</span></div>
                            </motion.div>
                        )}
                    </motion.div>
                </div>
            </section>

            {/* API Tester */}
            <section className="w-full max-w-[860px] mx-auto px-6 pb-14 relative z-10">
                <ApiTester />
            </section>

            {/* Quick Start */}
            <section className="w-full max-w-[860px] mx-auto px-6 pb-14 relative z-10">
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.6 }}>
                    <div className="flex items-center gap-3 mb-5">
                        <div className="p-1.5 rounded-lg bg-accent/10 text-accent"><BookOpen className="w-4 h-4" /></div>
                        <h2 className="text-heading font-semibold text-foreground">Quick Start</h2>
                    </div>
                    <div className="flex p-1 rounded-lg mb-4 w-fit border" style={{ background: 'hsl(var(--muted) / 0.4)', borderColor: 'hsl(var(--border) / 0.5)' }}>
                        {(["rest", "python", "js"] as const).map((tab) => (
                            <button key={tab} onClick={() => setActiveTab(tab)}
                                className={`px-4 py-1.5 rounded-md text-[13px] font-medium transition-all duration-200 ${activeTab === tab ? "bg-card shadow-soft text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                                {tab === "rest" ? "cURL" : tab === "python" ? "Python" : "JavaScript"}
                            </button>
                        ))}
                    </div>
                    <CodeBlock code={quickstartCode[activeTab]} language={activeTab === "rest" ? "bash" : activeTab} />
                </motion.div>
            </section>

            {/* Endpoints */}
            <section className="w-full max-w-[860px] mx-auto px-6 pb-32 relative z-10">
                <div className="flex items-center gap-3 mb-7">
                    <div className="p-1.5 rounded-lg bg-accent/10 text-accent"><Key className="w-4 h-4" /></div>
                    <h2 className="text-heading font-semibold text-foreground">All Endpoints ({endpoints.length})</h2>
                </div>

                <div className="space-y-5">
                    {endpoints.map((ep, i) => (
                        <motion.div key={i} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                            transition={{ duration: 0.4, delay: i * 0.03 }} className="card-elevated rounded-xl overflow-hidden">
                            <div className="px-5 py-4" style={{ borderBottom: "1px solid hsl(var(--border) / 0.3)" }}>
                                <div className="flex items-center gap-2 mb-1 flex-wrap">
                                    <span className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase tracking-wider ${ep.method === "POST" ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : ep.method === "GET" ? "bg-blue-500/10 text-blue-600 dark:text-blue-400" : "bg-orange-500/10 text-orange-600 dark:text-orange-400"}`}>
                                        {ep.method}
                                    </span>
                                    <code className="text-[14px] font-mono font-semibold text-foreground">{ep.path}</code>
                                    <span className="text-[10px] text-muted-foreground/50 font-mono ml-auto">{ep.source}</span>
                                </div>
                                <p className="text-caption text-muted-foreground">{ep.description}</p>
                            </div>

                            {ep.params.length > 0 && (
                                <div className="px-5 py-3" style={{ borderBottom: "1px solid hsl(var(--border) / 0.3)" }}>
                                    <h4 className="text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground mb-2">Parameters</h4>
                                    <div className="space-y-1.5">
                                        {ep.params.map(p => (
                                            <div key={p.name} className="flex items-center gap-2 text-[12px]">
                                                <code className="font-mono font-semibold text-foreground">{p.name}</code>
                                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground font-mono">{p.type}</span>
                                                {p.required && <span className="text-[10px] text-destructive font-bold">required</span>}
                                                <span className="text-muted-foreground/60">— {p.desc}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div className="p-4">
                                <h4 className="text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground mb-2 px-1">Example</h4>
                                <CodeBlock code={ep.curlExample} language="bash" />
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* Swagger link */}
                <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mt-12 text-center space-y-4">
                    <p className="text-caption text-muted-foreground">Full interactive Swagger UI available at:</p>
                    <div className="flex flex-wrap justify-center gap-3">
                        <a href={`${API_BASE}/swagger-ui/index.html`} target="_blank" rel="noopener noreferrer" className="btn-secondary text-[13px] px-5 py-2">
                            Open Swagger UI <ArrowRight className="w-3.5 h-3.5" />
                        </a>
                        <a href={`${ML_BASE}/docs`} target="_blank" rel="noopener noreferrer" className="btn-secondary text-[13px] px-5 py-2">
                            ML FastAPI Docs <ArrowRight className="w-3.5 h-3.5" />
                        </a>
                    </div>
                    <p className="text-body text-muted-foreground pt-4">Need API keys for production access?</p>
                    <Link href="/pricing" className="btn-primary text-[14px] px-6 py-2.5 inline-flex">
                        View Pricing Plans <ArrowRight className="w-3.5 h-3.5" />
                    </Link>
                </motion.div>
            </section>
        </div>
    );
}
