"use client";

import { Search, Filter, ArrowRight, Loader2 } from "lucide-react";
import { ResultBadge } from "@/components/shared/ResultBadge";
import Link from "next/link";
import { useState, useEffect } from "react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";

export default function HistoryPage() {
    const [filter, setFilter] = useState("All");
    const [history, setHistory] = useState<DetectionHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");

    useEffect(() => {
        async function fetchHistory() {
            try {
                setLoading(true);
                const data = await detectionApi.getHistory();
                setHistory(data);
            } catch (err) {
                console.error("Failed to fetch history:", err);
                setError("Failed to load scan history. Make sure the backend is running.");
            } finally {
                setLoading(false);
            }
        }
        fetchHistory();
    }, []);

    // Map backend resultLabel to display status
    const mapResult = (label: string): "Real" | "AI Generated" | "Suspicious" => {
        if (label.toLowerCase() === "fake") return "AI Generated";
        if (label.toLowerCase() === "real") return "Real";
        return "Suspicious";
    };

    // Filter and search
    const filteredHistory = history.filter((item) => {
        const displayResult = mapResult(item.resultLabel);
        const matchesFilter = filter === "All" || displayResult === filter;
        const matchesSearch = searchQuery === "" ||
            item.scanId.toString().includes(searchQuery) ||
            item.imagePath?.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesFilter && matchesSearch;
    });

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 md:p-8 max-w-7xl mx-auto space-y-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Scan History</h1>
                    <p className="text-muted-foreground mt-1">Review your previously analyzed images and reports.</p>
                </div>

                <div className="flex items-center gap-4 w-full md:w-auto">
                    <div className="relative flex-1 md:w-64">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                        <input
                            type="text"
                            placeholder="Search by ID or filename..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-4 py-2 rounded-full border bg-background focus:outline-none focus:ring-2 focus:ring-primary/20 transition-shadow"
                        />
                    </div>
                    <button className="flex items-center gap-2 px-4 py-2 border rounded-full bg-card hover:bg-muted transition-colors font-medium text-sm">
                        <Filter className="w-4 h-4" /> Filter
                    </button>
                </div>
            </div>

            <div className="flex flex-wrap gap-2 mb-6">
                {["All", "AI Generated", "Real", "Suspicious"].map(f => (
                    <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${filter === f ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-muted/80"}`}
                    >
                        {f}
                    </button>
                ))}
            </div>

            {loading && (
                <div className="flex flex-col items-center justify-center py-20 space-y-4">
                    <Loader2 className="w-8 h-8 text-primary animate-spin" />
                    <p className="text-muted-foreground">Loading scan history...</p>
                </div>
            )}

            {error && (
                <div className="p-6 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-center">
                    <p className="font-medium">{error}</p>
                </div>
            )}

            {!loading && !error && (
                <div className="bg-card border rounded-[2rem] overflow-hidden shadow-sm">
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-muted/50 text-muted-foreground font-semibold uppercase tracking-wider text-xs">
                                <tr>
                                    <th className="px-6 py-4">Scan ID</th>
                                    <th className="px-6 py-4">File</th>
                                    <th className="px-6 py-4">Result</th>
                                    <th className="px-6 py-4 hidden sm:table-cell">Confidence</th>
                                    <th className="px-6 py-4 hidden md:table-cell">Date Scanned</th>
                                    <th className="px-6 py-4 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border">
                                {filteredHistory.length === 0 ? (
                                    <tr>
                                        <td colSpan={6} className="px-6 py-12 text-center text-muted-foreground">
                                            {history.length === 0 ? "No scans yet. Go to the Detect page to analyze an image." : "No results match your filter."}
                                        </td>
                                    </tr>
                                ) : (
                                    filteredHistory.map((row) => {
                                        const displayResult = mapResult(row.resultLabel);
                                        const conf = row.confidenceScore ?? 0;
                                        return (
                                            <tr key={row.scanId} className="hover:bg-muted/20 transition-colors group">
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <span className="font-mono text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                                                        #{row.scanId}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <span className="text-sm truncate max-w-[200px] inline-block">{row.imagePath || "—"}</span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <ResultBadge status={displayResult} />
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap hidden sm:table-cell">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                                                            <div className={`h-full ${conf > 75 ? 'bg-destructive' : conf > 40 ? 'bg-yellow-500' : 'bg-green-500'}`} style={{ width: `${conf}%` }} />
                                                        </div>
                                                        <span className="font-medium">{conf.toFixed(1)}%</span>
                                                    </div>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-muted-foreground hidden md:table-cell">
                                                    {row.scanTimestamp ? new Date(row.scanTimestamp).toLocaleString() : "—"}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                                    <Link href={`/report/${row.scanId}`} className="inline-flex items-center gap-1 text-primary hover:text-primary/80 font-medium transition-colors">
                                                        View Report <ArrowRight className="w-4 h-4" />
                                                    </Link>
                                                </td>
                                            </tr>
                                        );
                                    })
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
