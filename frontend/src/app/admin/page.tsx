"use client";

import Link from "next/link";
import { ScanText, Users, AlertTriangle, ShieldAlert, Loader2, TrendingUp, Fingerprint, History, Eye, FileText, ArrowRight } from "lucide-react";
import { ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell } from "recharts";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";

const ease = [0.16, 1, 0.3, 1];

export default function AdminPage() {
    const [history, setHistory] = useState<DetectionHistoryItem[]>([]);
    const [stats, setStats] = useState<{ totalScans: number; totalUsers: number; deepfakesDetected: number } | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        async function fetchData() {
            try {
                setLoading(true);
                const [historyData, statsData] = await Promise.all([
                    detectionApi.getHistory(),
                    detectionApi.getStats(),
                ]);
                setHistory(historyData);
                setStats(statsData);
            } catch (err) {
                console.error("Failed to fetch admin data:", err);
                setError("Failed to load live metrics. Backend might be offline.");
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, []);

    // Compute live stats
    const totalScans = stats?.totalScans ?? history.length;
    const totalUsers = stats?.totalUsers ?? 0;
    const deepfakesDetected = stats?.deepfakesDetected ?? history.filter(h => h.resultLabel.toLowerCase() === "fake").length;
    const fakeRate = totalScans > 0 ? Math.round((deepfakesDetected / totalScans) * 100) : 0;

    const statCards = [
        { label: "Total Scans", value: totalScans.toLocaleString(), change: "Live", changeType: "up" as const, icon: ScanText, iconColor: "text-accent" },
        { label: "Fake Rate", value: `${fakeRate}%`, change: `${deepfakesDetected.toLocaleString()} detected`, changeType: "up" as const, icon: AlertTriangle, iconColor: "text-destructive" },
        { label: "Registered Users", value: totalUsers.toLocaleString(), change: "Live", changeType: "up" as const, icon: Users, iconColor: "text-blue-500" },
        { label: "Deepfakes Detected", value: deepfakesDetected.toLocaleString(), change: "Live", changeType: "up" as const, icon: ShieldAlert, iconColor: "text-orange-500" },
    ];

    // Generate last 30 days of date-based chart data
    const generateTimelineData = () => {
        const now = new Date();
        const daysToShow = 30;
        const dayMap: Record<string, { scans: number; fakes: number; real: number }> = {};

        // Initialize last 30 days
        for (let i = daysToShow - 1; i >= 0; i--) {
            const d = new Date(now);
            d.setDate(d.getDate() - i);
            const key = d.toISOString().split("T")[0];
            dayMap[key] = { scans: 0, fakes: 0, real: 0 };
        }

        // Populate from history
        history.forEach(item => {
            if (!item.scanTimestamp) return;
            const key = new Date(item.scanTimestamp).toISOString().split("T")[0];
            if (dayMap[key]) {
                dayMap[key].scans++;
                if (item.resultLabel.toLowerCase() === "fake") {
                    dayMap[key].fakes++;
                } else {
                    dayMap[key].real++;
                }
            }
        });

        return Object.entries(dayMap).map(([date, data]) => {
            const d = new Date(date);
            return {
                date,
                label: d.toLocaleDateString("en-IN", { day: "numeric", month: "short" }),
                ...data,
            };
        });
    };

    const timelineData = generateTimelineData();

    // Pie chart data for verdict breakdown
    const realCount = totalScans - deepfakesDetected;
    const pieData = [
        { name: "Real", value: realCount, color: "#22c55e" },
        { name: "Fake", value: deepfakesDetected, color: "#ef4444" },
    ];

    // Custom tooltip
    const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; dataKey: string; color: string }>; label?: string }) => {
        if (!active || !payload) return null;
        return (
            <div className="rounded-xl px-4 py-3 shadow-lg border text-[13px]"
                style={{ background: "hsl(var(--card))", borderColor: "hsl(var(--border))" }}>
                <p className="font-semibold text-foreground mb-1.5">{label}</p>
                {payload.map((p, i) => (
                    <div key={i} className="flex items-center gap-2 py-0.5">
                        <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
                        <span className="text-muted-foreground capitalize">{p.dataKey}:</span>
                        <span className="font-semibold font-mono">{p.value}</span>
                    </div>
                ))}
            </div>
        );
    };

    if (!mounted) {
        return (
            <div className="min-h-[calc(100vh-4rem)] max-w-[1120px] mx-auto px-6 py-10 flex flex-col items-center justify-center space-y-3">
                <Loader2 className="w-6 h-6 text-accent animate-spin" />
            </div>
        );
    }

    return (
        <div className="min-h-[calc(100vh-4rem)] max-w-[1120px] mx-auto px-6 py-10 space-y-7 relative">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 relative z-10">
                <div>
                    <h1 className="text-heading-lg font-serif text-foreground">System Overview</h1>
                    <p className="text-caption text-muted-foreground mt-1">Live global verification metrics and threat intelligence.</p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] font-medium" style={{ background: "hsl(var(--accent) / 0.08)", color: "hsl(var(--accent))" }}>
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                    Live Analysis Server
                </div>
            </div>

            {loading ? (
                <div className="flex flex-col items-center justify-center py-24 space-y-3 relative z-10">
                    <Loader2 className="w-6 h-6 text-accent animate-spin" />
                    <p className="text-caption text-muted-foreground">Loading live dashboard metrics...</p>
                </div>
            ) : error ? (
                <div className="p-5 rounded-xl bg-destructive/5 border border-destructive/15 text-center relative z-10">
                    <p className="text-[14px] font-medium text-destructive">{error}</p>
                </div>
            ) : (
                <>
                    {/* Stat Cards */}
                    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, ease }} className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
                        {statCards.map((stat, i) => (
                            <div key={i} className="card-elevated rounded-xl p-5 flex flex-col justify-between min-h-[130px]">
                                <div className="flex justify-between items-start">
                                    <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">{stat.label}</span>
                                    <stat.icon className={`w-4 h-4 ${stat.iconColor}`} strokeWidth={1.8} />
                                </div>
                                <div>
                                    <div className="text-3xl font-bold font-mono tracking-tight text-foreground">{stat.value}</div>
                                    <div className={`text-[12px] font-medium mt-0.5 ${stat.label === "Fake Rate" ? "text-destructive" : "text-green-500"}`}>
                                        {stat.change}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </motion.div>

                    {/* Main Charts Row */}
                    <div className="grid lg:grid-cols-3 gap-5 relative z-10">
                        {/* Bar + Line combo chart — spans 2 columns */}
                        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1, duration: 0.5, ease }}
                            className="lg:col-span-2 card-elevated rounded-xl p-6 lg:p-7">
                            <div className="flex items-center justify-between mb-5">
                                <div>
                                    <h2 className="text-[16px] font-semibold text-foreground">Scan Activity — Last 30 Days</h2>
                                    <p className="text-[12px] text-muted-foreground mt-0.5">Daily scan volume with deepfake detection overlay</p>
                                </div>
                                <div className="flex items-center gap-1.5 text-[12px] text-green-500 font-medium">
                                    <TrendingUp className="w-3.5 h-3.5" /> Live
                                </div>
                            </div>
                            <div className="h-[340px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={timelineData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                                        <defs>
                                            <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="hsl(var(--accent))" stopOpacity={0.85} />
                                                <stop offset="100%" stopColor="hsl(var(--accent))" stopOpacity={0.4} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.4)" vertical={false} />
                                        <XAxis
                                            dataKey="label"
                                            stroke="hsl(var(--muted-foreground))"
                                            fontSize={11}
                                            tickLine={false}
                                            axisLine={false}
                                            interval={Math.floor(timelineData.length / 8)}
                                            angle={-30}
                                            textAnchor="end"
                                            height={50}
                                        />
                                        <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} allowDecimals={false} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Legend
                                            verticalAlign="top"
                                            align="right"
                                            height={36}
                                            iconType="circle"
                                            iconSize={8}
                                            formatter={(value: string) => <span className="text-[12px] text-muted-foreground capitalize ml-1">{value}</span>}
                                        />
                                        <Bar dataKey="scans" fill="url(#barGrad)" radius={[4, 4, 0, 0]} barSize={timelineData.length > 15 ? 12 : 20} name="scans" />
                                        <Line type="monotone" dataKey="fakes" stroke="#ef4444" strokeWidth={2.5} dot={false} activeDot={{ r: 4, fill: "#ef4444", strokeWidth: 0 }} name="fakes" />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </motion.div>

                        {/* Pie chart + breakdown — single column */}
                        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2, duration: 0.5, ease }}
                            className="card-elevated rounded-xl p-6 lg:p-7 flex flex-col">
                            <h2 className="text-[16px] font-semibold text-foreground mb-1">Verdict Breakdown</h2>
                            <p className="text-[12px] text-muted-foreground mb-4">Overall detection distribution</p>

                            <div className="flex-1 flex items-center justify-center">
                                <div className="w-[200px] h-[200px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={pieData}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={55}
                                                outerRadius={85}
                                                paddingAngle={4}
                                                dataKey="value"
                                                strokeWidth={0}
                                            >
                                                {pieData.map((entry, i) => (
                                                    <Cell key={i} fill={entry.color} />
                                                ))}
                                            </Pie>
                                            <Tooltip
                                                contentStyle={{
                                                    background: "hsl(var(--card))",
                                                    border: "1px solid hsl(var(--border))",
                                                    borderRadius: "0.75rem",
                                                    fontSize: "13px",
                                                }}
                                                formatter={(value: number, name: string) => [`${value.toLocaleString()}`, name]}
                                            />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="space-y-3 mt-4 pt-4 border-t" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                {pieData.map((item) => (
                                    <div key={item.name} className="flex items-center justify-between">
                                        <div className="flex items-center gap-2.5">
                                            <span className="w-2.5 h-2.5 rounded-full" style={{ background: item.color }} />
                                            <span className="text-[13px] text-muted-foreground">{item.name}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-[14px] font-semibold font-mono">{item.value.toLocaleString()}</span>
                                            <span className="text-[11px] text-muted-foreground">
                                                ({totalScans > 0 ? Math.round((item.value / totalScans) * 100) : 0}%)
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    </div>

                    {/* Quick Actions + Recent Scans */}
                    <div className="grid lg:grid-cols-3 gap-5 relative z-10">
                        {/* Quick Actions */}
                        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3, duration: 0.5, ease }}
                            className="card-elevated rounded-xl p-6 lg:p-7 space-y-4">
                            <h2 className="text-[16px] font-semibold text-foreground">Quick Actions</h2>
                            <div className="space-y-2">
                                {[
                                    { icon: ScanText, label: "Image Detection", desc: "Analyze an image", href: "/detect", color: "text-accent" },
                                    { icon: Fingerprint, label: "Identity Protection", desc: "Reverse image search", href: "/protection", color: "text-violet-500" },
                                    { icon: History, label: "Scan History", desc: "View past results", href: "/history", color: "text-blue-500" },
                                ].map((action) => (
                                    <Link key={action.label} href={action.href}
                                        className="flex items-center gap-3 p-3 rounded-xl hover:bg-muted/50 transition-colors group">
                                        <div className={`p-2 rounded-lg bg-muted/50 ${action.color} group-hover:bg-accent/10 transition-colors`}>
                                            <action.icon className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="text-[13px] font-semibold text-foreground">{action.label}</div>
                                            <div className="text-[11px] text-muted-foreground">{action.desc}</div>
                                        </div>
                                        <ArrowRight className="w-3.5 h-3.5 text-muted-foreground/50 group-hover:text-accent transition-colors" />
                                    </Link>
                                ))}
                            </div>
                        </motion.div>

                        {/* Recent Scans */}
                        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.35, duration: 0.5, ease }}
                            className="lg:col-span-2 card-elevated rounded-xl p-6 lg:p-7 space-y-4">
                            <div className="flex items-center justify-between">
                                <h2 className="text-[16px] font-semibold text-foreground">Recent Scans</h2>
                                <Link href="/history" className="text-[12px] text-accent font-medium hover:underline flex items-center gap-1">
                                    View All <ArrowRight className="w-3 h-3" />
                                </Link>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="w-full text-[13px]">
                                    <thead>
                                        <tr className="border-b" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                            <th className="text-left py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">ID</th>
                                            <th className="text-left py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">File</th>
                                            <th className="text-left py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Verdict</th>
                                            <th className="text-left py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Confidence</th>
                                            <th className="text-left py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Date</th>
                                            <th className="text-right py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Report</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {history.slice(0, 5).map((scan) => (
                                            <tr key={scan.scanId} className="border-b last:border-0 hover:bg-muted/30 transition-colors" style={{ borderColor: "hsl(var(--border) / 0.25)" }}>
                                                <td className="py-2.5 font-mono font-medium">#{scan.scanId}</td>
                                                <td className="py-2.5 text-muted-foreground truncate max-w-[120px]">{scan.imagePath}</td>
                                                <td className="py-2.5">
                                                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-semibold ${
                                                        scan.resultLabel.toLowerCase() === "fake"
                                                            ? "bg-destructive/8 text-destructive"
                                                            : "bg-green-500/8 text-green-600 dark:text-green-400"
                                                    }`}>
                                                        <span className={`w-1.5 h-1.5 rounded-full ${scan.resultLabel.toLowerCase() === "fake" ? "bg-destructive" : "bg-green-500"}`} />
                                                        {scan.resultLabel}
                                                    </span>
                                                </td>
                                                <td className="py-2.5 font-mono">{scan.confidenceScore?.toFixed(1)}%</td>
                                                <td className="py-2.5 text-muted-foreground">
                                                    {scan.scanTimestamp ? new Date(scan.scanTimestamp).toLocaleDateString("en-IN", { day: "numeric", month: "short" }) : "—"}
                                                </td>
                                                <td className="py-2.5 text-right">
                                                    <Link href={`/report/${scan.scanId}`}
                                                        className="inline-flex items-center gap-1 text-accent hover:underline text-[12px] font-medium">
                                                        <FileText className="w-3 h-3" /> View
                                                    </Link>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </motion.div>
                    </div>
                </>
            )}
        </div>
    );
}
