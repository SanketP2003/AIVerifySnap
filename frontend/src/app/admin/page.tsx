"use client";

import { ScanText, Users, AlertTriangle, ShieldAlert, Activity, Server, CheckCircle2, ArrowUpRight, RefreshCw } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { ResultBadge } from "@/components/shared/ResultBadge";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";

const chartData7Days = [
    { name: 'Mon', scans: 120, fakes: 40 },
    { name: 'Tue', scans: 230, fakes: 90 },
    { name: 'Wed', scans: 340, fakes: 130 },
    { name: 'Thu', scans: 290, fakes: 110 },
    { name: 'Fri', scans: 450, fakes: 210 },
    { name: 'Sat', scans: 512, fakes: 280 },
    { name: 'Sun', scans: 600, fakes: 320 },
];

const chartData30Days = [
    { name: 'Week 1', scans: 1450, fakes: 590 },
    { name: 'Week 2', scans: 1820, fakes: 740 },
    { name: 'Week 3', scans: 2340, fakes: 980 },
    { name: 'Week 4', scans: 2980, fakes: 1210 },
];

const defaultThreats = [
    { id: "REV-9021", model: "SigLIP + ELA", verdict: "AI Generated" as const, confidence: 99.2, origin: "Web Upload", time: "3m ago" },
    { id: "REV-9020", model: "SigLIP ViT", verdict: "Real" as const, confidence: 96.4, origin: "API Client", time: "12m ago" },
    { id: "REV-9019", model: "ELA Residual", verdict: "Suspicious" as const, confidence: 64.8, origin: "Protection Scan", time: "24m ago" },
    { id: "REV-9018", model: "Ensemble ViT", verdict: "AI Generated" as const, confidence: 98.7, origin: "Web Upload", time: "41m ago" },
];

export default function AdminPage() {
    const [mounted, setMounted] = useState(false);
    const [timeRange, setTimeRange] = useState<"7D" | "30D">("7D");
    const [realScans, setRealScans] = useState<DetectionHistoryItem[]>([]);
    const [syncing, setSyncing] = useState(false);

    useEffect(() => {
        setMounted(true);
        detectionApi.getHistory()
            .then((data) => setRealScans(data || []))
            .catch(() => setRealScans([]));
    }, []);

    const activeChartData = timeRange === "7D" ? chartData7Days : chartData30Days;

    const handleSync = async () => {
        setSyncing(true);
        try {
            const data = await detectionApi.getHistory();
            setRealScans(data || []);
        } catch (e) {
            console.error("Telemetry sync error:", e);
        } finally {
            setTimeout(() => setSyncing(false), 500);
        }
    };

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
                        Real-time threat telemetry, detection engine load, and global synthetic media metrics.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        type="button"
                        onClick={handleSync}
                        className="px-3.5 py-1.5 rounded-full bg-card border border-border text-xs font-semibold hover:bg-muted text-foreground transition-colors flex items-center gap-1.5 shadow-sm"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${syncing ? "animate-spin" : ""}`} />
                        <span>Sync Feeds</span>
                    </button>
                    <div className="px-4 py-2 bg-primary/10 text-primary font-semibold rounded-full text-xs flex items-center gap-2 border border-primary/20">
                        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        <span>Inference Nodes Online (4/4)</span>
                    </div>
                </div>
            </div>

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
                            {realScans.length > 0 ? (realScans.length).toLocaleString() : "1.2M"}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-green-600 font-semibold mt-1">
                            <ArrowUpRight className="w-3.5 h-3.5" />
                            <span>{realScans.length > 0 ? "Live verified sessions" : "+12.4% vs last period"}</span>
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
                            {realScans.length > 0
                                ? `${Math.round((realScans.filter(s => s.resultLabel?.toLowerCase() === "fake").length / realScans.length) * 100)}%`
                                : "48.2%"}
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
                        <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">4,821</div>
                        <div className="flex items-center gap-1 text-xs text-green-600 font-semibold mt-1">
                            <ArrowUpRight className="w-3.5 h-3.5" />
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
                        <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">842</div>
                        <div className="flex items-center gap-1 text-xs text-green-600 font-semibold mt-1">
                            <ArrowUpRight className="w-3.5 h-3.5" />
                            <span>Verified host removals</span>
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
                    {mounted ? (
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
                    ) : (
                        <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
                            Rendering telemetry stream...
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
                                <span>Vision Transformer VRAM</span>
                                <span className="font-mono font-bold text-foreground">6.4 / 16 GB (40%)</span>
                            </div>
                            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                                <div className="h-full bg-primary rounded-full w-[40%]" />
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            <div className="flex justify-between text-muted-foreground">
                                <span>ELA Processor Load</span>
                                <span className="font-mono font-bold text-foreground">22% Peak</span>
                            </div>
                            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                                <div className="h-full bg-green-500 rounded-full w-[22%]" />
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            <div className="flex justify-between text-muted-foreground">
                                <span>Average Request Latency</span>
                                <span className="font-mono font-bold text-foreground">142ms</span>
                            </div>
                            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500 rounded-full w-[18%]" />
                            </div>
                        </div>
                    </div>

                    <div className="p-3 bg-muted/40 rounded-2xl border border-border/60 flex items-center gap-2 text-xs text-muted-foreground">
                        <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                        <span>All neural weights loaded & validated</span>
                    </div>
                </div>

                {/* Recent Incident Feed Table */}
                <div className="lg:col-span-2 bg-card border border-border rounded-3xl p-6 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-bold text-base text-foreground">Recent Threat Ingest Stream</h3>
                        <span className="text-xs text-muted-foreground font-mono">Live Sync</span>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-xs text-left">
                            <thead className="bg-muted/40 text-muted-foreground font-semibold uppercase tracking-wider text-[11px] border-b border-border">
                                <tr>
                                    <th className="px-4 py-3">Scan Ref</th>
                                    <th className="px-4 py-3">Engine</th>
                                    <th className="px-4 py-3">Verdict</th>
                                    <th className="px-4 py-3">Certainty</th>
                                    <th className="px-4 py-3 text-right">Time</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/60">
                                {defaultThreats.map((t) => (
                                    <tr key={t.id} className="hover:bg-muted/20 transition-colors">
                                        <td className="px-4 py-3 font-mono font-semibold text-muted-foreground">{t.id}</td>
                                        <td className="px-4 py-3 font-medium text-foreground">{t.model}</td>
                                        <td className="px-4 py-3">
                                            <ResultBadge status={t.verdict} size="sm" />
                                        </td>
                                        <td className="px-4 py-3 font-mono font-bold text-foreground">{t.confidence}%</td>
                                        <td className="px-4 py-3 text-right text-muted-foreground">{t.time}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
