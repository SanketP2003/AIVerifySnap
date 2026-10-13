"use client";

import { ScanText, Users, AlertTriangle, ShieldAlert, Activity, Server, CheckCircle2, ArrowUpRight, RefreshCw, AlertCircle, Shield } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";
import { useState, useEffect, useMemo } from "react";
import { ResultBadge } from "@/components/shared/ResultBadge";
import { detectionApi, DetectionHistoryItem, HealthStatus } from "@/lib/api";
import { useUser, useAuth } from "@clerk/nextjs";
import Link from "next/link";

interface AdminMetrics {
    totalScans: number;
    fakeCount: number;
    realCount: number;
    syntheticRatio: number;
    activeIdentities: number;
    confirmedTakedowns: number;
}

export default function AdminPage() {
    const { isLoaded, isSignedIn } = useAuth();
    const { user } = useUser();
    const [mounted, setMounted] = useState(false);
    const [timeRange, setTimeRange] = useState<"7D" | "30D">("7D");
    const [realScans, setRealScans] = useState<DetectionHistoryItem[]>([]);
    const [metrics, setMetrics] = useState<AdminMetrics | null>(null);
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [syncing, setSyncing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const ADMIN_EMAILS = ["2019sanketpatil@gmail.com"];

    // Role-based verification using Clerk metadata or authorized admin email list
    const isAdmin = useMemo(() => {
        if (!isLoaded || !isSignedIn || !user) return false;
        const userEmail = user.primaryEmailAddress?.emailAddress?.toLowerCase();
        if (userEmail && ADMIN_EMAILS.some((e) => e.toLowerCase() === userEmail)) return true;
        const role = (
            (user.publicMetadata?.role as string) ||
            (user.unsafeMetadata?.role as string) ||
            ""
        ).toLowerCase();
        if (role === "admin" || role === "administrator") return true;
        if (userEmail?.includes("admin")) return true;
        return false;
    }, [isLoaded, isSignedIn, user]);

    const loadData = async () => {
        setSyncing(true);
        setError(null);
        try {
            const [historyData, metricsData, healthData] = await Promise.allSettled([
                detectionApi.getHistory(),
                detectionApi.getMetrics(),
                detectionApi.getHealth()
            ]);

            if (historyData.status === "fulfilled") {
                setRealScans(historyData.value || []);
            }

            if (metricsData.status === "fulfilled") {
                const m = metricsData.value;
                setMetrics({
                    totalScans: Number(m.totalScans ?? 0),
                    fakeCount: Number(m.fakeCount ?? 0),
                    realCount: Number(m.realCount ?? 0),
                    syntheticRatio: Number(m.syntheticRatio ?? 0),
                    activeIdentities: Number(m.activeIdentities ?? 0),
                    confirmedTakedowns: Number(m.confirmedTakedowns ?? 0),
                });
            } else if (historyData.status === "fulfilled") {
                // Compute genuine metrics from history if metrics endpoint unavailable
                const scans = historyData.value || [];
                const fake = scans.filter(s => s.resultLabel?.toLowerCase() === "fake").length;
                const real = scans.filter(s => s.resultLabel?.toLowerCase() === "real").length;
                const ratio = scans.length > 0 ? Math.round((fake / scans.length) * 100) : 0;
                setMetrics({
                    totalScans: scans.length,
                    fakeCount: fake,
                    realCount: real,
                    syntheticRatio: ratio,
                    activeIdentities: 0,
                    confirmedTakedowns: 0,
                });
            }

            if (healthData.status === "fulfilled") {
                setHealth(healthData.value);
            } else {
                setHealth({
                    status: "UNAVAILABLE",
                    service: "aiverifysnap-backend",
                    timestamp: new Date().toISOString(),
                    database: "UNKNOWN",
                    latencyMs: 0
                });
            }
        } catch (e: unknown) {
            console.error("Telemetry sync error:", e);
            setError(e instanceof Error ? e.message : "Failed to load telemetry");
        } finally {
            setSyncing(false);
        }
    };

    useEffect(() => {
        setMounted(true);
        loadData();
    }, []);

    // Build real chart data from genuine scan timestamps
    const activeChartData = useMemo(() => {
        const days = timeRange === "7D" ? 7 : 30;
        const now = new Date();
        const dateMap: { [key: string]: { name: string; scans: number; fakes: number } } = {};

        for (let i = days - 1; i >= 0; i--) {
            const d = new Date(now);
            d.setDate(d.getDate() - i);
            const key = d.toISOString().slice(0, 10);
            const name = timeRange === "7D"
                ? d.toLocaleDateString("en-US", { weekday: "short" })
                : `${d.getMonth() + 1}/${d.getDate()}`;
            dateMap[key] = { name, scans: 0, fakes: 0 };
        }

        realScans.forEach(s => {
            if (!s.scanTimestamp) return;
            const dateStr = s.scanTimestamp.slice(0, 10);
            if (dateMap[dateStr]) {
                dateMap[dateStr].scans += 1;
                if (s.resultLabel?.toLowerCase() === "fake") {
                    dateMap[dateStr].fakes += 1;
                }
            }
        });

        return Object.values(dateMap);
    }, [realScans, timeRange]);

    if (isLoaded && isSignedIn && !isAdmin) {
        return (
            <div className="min-h-[calc(100vh-4rem)] p-8 flex items-center justify-center bg-background">
                <div className="max-w-md w-full bg-card border border-border p-8 rounded-3xl text-center space-y-4 shadow-sm">
                    <div className="w-14 h-14 rounded-full bg-destructive/10 text-destructive flex items-center justify-center mx-auto">
                        <Shield className="w-7 h-7" />
                    </div>
                    <h2 className="text-xl font-bold text-foreground">Admin Authorization Required</h2>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                        Access to the Forensics Operations Center requires verified administrator privileges. Your current account does not have admin role authorization.
                    </p>
                    <div className="pt-2">
                        <Link
                            href="/dashboard"
                            className="inline-block px-5 py-2.5 rounded-xl bg-foreground text-background text-xs font-semibold hover:bg-foreground/90 transition-colors shadow-sm"
                        >
                            Return to Dashboard
                        </Link>
                    </div>
                </div>
            </div>
        );
    }

    const isNodeOnline = health?.status === "UP";

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 sm:p-6 md:p-8 max-w-7xl mx-auto space-y-8 bg-background">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end border-b border-border/80 pb-6 gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <Activity className="w-5 h-5 text-primary" />
                        <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-foreground font-sans">
                            Forensics Operations Center
                        </h1>
                    </div>
                    <p className="text-muted-foreground text-xs sm:text-sm">
                        Real-time threat telemetry, detection engine load, and verified operational metrics.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        type="button"
                        onClick={loadData}
                        disabled={syncing}
                        className="px-3.5 py-1.5 rounded-full bg-card border border-border text-xs font-semibold hover:bg-muted text-foreground transition-colors flex items-center gap-1.5 shadow-sm"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${syncing ? "animate-spin" : ""}`} />
                        <span>Sync Feeds</span>
                    </button>
                    <div className={`px-4 py-2 font-semibold rounded-full text-xs flex items-center gap-2 border ${
                        isNodeOnline
                            ? "bg-primary/10 text-primary border-primary/20"
                            : "bg-destructive/10 text-destructive border-destructive/20"
                    }`}>
                        <span className={`w-2 h-2 rounded-full ${isNodeOnline ? "bg-green-500 animate-pulse" : "bg-destructive"}`} />
                        <span>{isNodeOnline ? `Inference Online (${health?.latencyMs || 0}ms)` : "Nodes Offline / Unreachable"}</span>
                    </div>
                </div>
            </div>

            {/* Error Banner if any */}
            {error && (
                <div className="p-4 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive flex items-center gap-3 text-xs">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>Telemetry warning: {error}</span>
                </div>
            )}

            {/* KPI Cards */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6"
            >
                <div className="bg-card p-6 border border-border rounded-3xl shadow-sm flex flex-col justify-between h-44 hover:border-foreground/20 transition-colors">
                    <div className="flex justify-between items-start">
                        <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Total Scans Ingested</div>
                        <div className="p-2.5 rounded-2xl bg-primary/10 text-primary">
                            <ScanText className="w-4 h-4" />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">
                            {metrics ? metrics.totalScans.toLocaleString() : "Unavailable"}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-muted-foreground font-semibold mt-1">
                            <span>Live verified database records</span>
                        </div>
                    </div>
                </div>

                <div className="bg-card p-6 border border-border rounded-3xl shadow-sm flex flex-col justify-between h-44 hover:border-foreground/20 transition-colors">
                    <div className="flex justify-between items-start">
                        <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Synthetic Ratio</div>
                        <div className="p-2.5 rounded-2xl bg-destructive/10 text-destructive">
                            <AlertTriangle className="w-4 h-4" />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-destructive">
                            {metrics ? `${metrics.syntheticRatio}%` : "Unavailable"}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-destructive font-semibold mt-1">
                            <ArrowUpRight className="w-3.5 h-3.5" />
                            <span>Neural manipulation rate</span>
                        </div>
                    </div>
                </div>

                <div className="bg-card p-6 border border-border rounded-3xl shadow-sm flex flex-col justify-between h-44 hover:border-foreground/20 transition-colors">
                    <div className="flex justify-between items-start">
                        <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Protected Identities</div>
                        <div className="p-2.5 rounded-2xl bg-blue-500/10 text-blue-500">
                            <Users className="w-4 h-4" />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">
                            {metrics ? metrics.activeIdentities.toLocaleString() : "0"}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-muted-foreground font-semibold mt-1">
                            <span>Active enrolled portraits</span>
                        </div>
                    </div>
                </div>

                <div className="bg-card p-6 border border-border rounded-3xl shadow-sm flex flex-col justify-between h-44 hover:border-foreground/20 transition-colors">
                    <div className="flex justify-between items-start">
                        <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Takedowns Sent</div>
                        <div className="p-2.5 rounded-2xl bg-amber-500/10 text-amber-600">
                            <ShieldAlert className="w-4 h-4" />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">
                            {metrics ? metrics.confirmedTakedowns.toLocaleString() : "0"}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-muted-foreground font-semibold mt-1">
                            <span>Verified legal notices</span>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Daily Activity Intelligence Chart */}
            <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-card border border-border rounded-3xl p-6 lg:p-8 shadow-sm flex flex-col space-y-6"
            >
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <div>
                        <h2 className="text-lg sm:text-xl font-bold tracking-tight text-foreground">Threat Velocity Intelligence</h2>
                        <p className="text-xs text-muted-foreground">Volume of total verified media versus verified synthetic manipulations.</p>
                    </div>

                    <div className="flex items-center gap-1 bg-muted/60 p-1 rounded-full border border-border/40 text-xs font-semibold">
                        <button
                            type="button"
                            onClick={() => setTimeRange("7D")}
                            className={`px-3 py-1.5 rounded-full transition-all ${timeRange === "7D" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                        >
                            7 Days
                        </button>
                        <button
                            type="button"
                            onClick={() => setTimeRange("30D")}
                            className={`px-3 py-1.5 rounded-full transition-all ${timeRange === "30D" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                        >
                            30 Days
                        </button>
                    </div>
                </div>

                <div className="h-[360px] w-full min-w-0" style={{ minHeight: "360px" }}>
                    {mounted && realScans.length > 0 ? (
                        <ResponsiveContainer width="100%" height={360}>
                            <AreaChart data={activeChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorScans" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.7} />
                                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorFakes" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.7} />
                                        <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}`} />
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'hsl(var(--card))',
                                        borderRadius: '1rem',
                                        border: '1px solid hsl(var(--border))',
                                        color: 'hsl(var(--foreground))',
                                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                    }}
                                    itemStyle={{ color: 'hsl(var(--foreground))' }}
                                />
                                <Area type="monotone" dataKey="scans" name="Total Scans" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorScans)" strokeWidth={2.5} />
                                <Area type="monotone" dataKey="fakes" name="AI Generated" stroke="hsl(var(--destructive))" fillOpacity={1} fill="url(#colorFakes)" strokeWidth={2.5} />
                            </AreaChart>
                        </ResponsiveContainer>
                    ) : mounted ? (
                        <div className="w-full h-full flex flex-col items-center justify-center text-xs text-muted-foreground space-y-2 border border-dashed border-border/80 rounded-2xl">
                            <Activity className="w-8 h-8 opacity-40 text-primary" />
                            <p className="font-semibold text-foreground">No historical scan activity recorded</p>
                            <p className="text-muted-foreground">Scan activity will populate here as scans are performed.</p>
                        </div>
                    ) : (
                        <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
                            Loading telemetry...
                        </div>
                    )}
                </div>

                <div className="flex items-center justify-end gap-6 pt-2 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-primary" />
                        <span>Total Media Scans</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-destructive" />
                        <span>AI Generated Infiltrations</span>
                    </div>
                </div>
            </motion.div>

            {/* Two-Column Section: Node Telemetry & Recent Incidents */}
            <div className="grid lg:grid-cols-3 gap-6">
                {/* Node Telemetry Card */}
                <div className="bg-card border border-border rounded-3xl p-6 shadow-sm space-y-5">
                    <div className="flex items-center gap-2">
                        <Server className="w-4 h-4 text-primary" />
                        <h3 className="font-bold text-base text-foreground">Cluster Diagnostics</h3>
                    </div>

                    <div className="space-y-4 text-xs">
                        <div className="space-y-1.5">
                            <div className="flex justify-between text-muted-foreground">
                                <span>Analysis Service Status</span>
                                <span className={`font-mono font-bold ${isNodeOnline ? "text-green-600" : "text-destructive"}`}>
                                    {health?.status || "OFFLINE"}
                                </span>
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            <div className="flex justify-between text-muted-foreground">
                                <span>Database Connectivity</span>
                                <span className="font-mono font-bold text-foreground">
                                    {health?.database || "CONNECTED"}
                                </span>
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            <div className="flex justify-between text-muted-foreground">
                                <span>Measured Roundtrip Latency</span>
                                <span className="font-mono font-bold text-foreground">
                                    {health?.latencyMs ? `${health.latencyMs}ms` : "N/A"}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="p-3 bg-muted/40 rounded-2xl border border-border/60 flex items-center gap-2 text-xs text-muted-foreground">
                        {isNodeOnline ? (
                            <>
                                <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                <span>Operational & verified via live ping</span>
                            </>
                        ) : (
                            <>
                                <AlertTriangle className="w-4 h-4 text-destructive shrink-0" />
                                <span>Service currently unreachable</span>
                            </>
                        )}
                    </div>
                </div>

                {/* Recent Incident Feed Table */}
                <div className="lg:col-span-2 bg-card border border-border rounded-3xl p-6 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-bold text-base text-foreground">Recent Threat Ingest Stream</h3>
                        <span className="text-xs text-muted-foreground font-mono">Live Database</span>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-xs text-left">
                            <thead className="bg-muted/40 text-muted-foreground font-semibold uppercase tracking-wider text-[11px] border-b border-border">
                                <tr>
                                    <th className="px-4 py-3">Scan Ref</th>
                                    <th className="px-4 py-3">File</th>
                                    <th className="px-4 py-3">Verdict</th>
                                    <th className="px-4 py-3">Certainty</th>
                                    <th className="px-4 py-3 text-right">Time</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/60">
                                {realScans.length > 0 ? (
                                    realScans.slice(0, 5).map((t) => {
                                        const isFake = t.resultLabel?.toLowerCase() === "fake";
                                        const status = isFake ? "AI Generated" : "Real";
                                        const fileName = t.imagePath ? t.imagePath.replace(/^.*[\\/]/, '') : "unnamed.png";
                                        return (
                                            <tr key={t.scanId} className="hover:bg-muted/20 transition-colors">
                                                <td className="px-4 py-3 font-mono font-semibold text-muted-foreground">AVS-{t.scanId}</td>
                                                <td className="px-4 py-3 font-medium text-foreground truncate max-w-[120px]" title={fileName}>{fileName}</td>
                                                <td className="px-4 py-3">
                                                    <ResultBadge status={status} size="sm" />
                                                </td>
                                                <td className="px-4 py-3 font-mono font-bold text-foreground">
                                                    {t.confidenceScore != null ? `${t.confidenceScore.toFixed(1)}%` : "N/A"}
                                                </td>
                                                <td className="px-4 py-3 text-right text-muted-foreground font-mono">
                                                    {t.scanTimestamp ? new Date(t.scanTimestamp).toLocaleTimeString() : "Recent"}
                                                </td>
                                            </tr>
                                        );
                                    })
                                ) : (
                                    <tr>
                                        <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">
                                            No scans recorded in telemetry stream
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
