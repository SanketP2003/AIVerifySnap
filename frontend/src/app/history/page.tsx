"use client";

import {
    Search,
    ArrowRight,
    Loader2,
    RefreshCw,
    X,
    Shield,
    Calendar,
    HardDrive,
    Sparkles,
    ArrowUpDown,
    ChevronLeft,
    ChevronRight,
    FileSpreadsheet,
    FileCode,
    Trash2
} from "lucide-react";
import { ResultBadge } from "@/components/shared/ResultBadge";
import Link from "next/link";
import { useState, useEffect, useCallback, useMemo } from "react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";
import { toast } from "sonner";

type SortOption = "newest" | "oldest" | "highest" | "lowest";

export default function HistoryPage() {
    const [filter, setFilter] = useState("All");
    const [history, setHistory] = useState<DetectionHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [sortBy, setSortBy] = useState<SortOption>("newest");
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 8;

    const fetchHistory = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await detectionApi.getHistory();
            setHistory(data || []);
        } catch (err) {
            console.error("Failed to fetch history:", err);
            setError("Failed to load scan history. Make sure the backend service is running.");
        } finally {
            setLoading(false);
        }
    }, []);

    const handleDeleteItem = async (scanId: number) => {
        try {
            await detectionApi.deleteHistoryItem(scanId);
            setHistory((prev) => prev.filter((item) => item.scanId !== scanId));
            toast.success(`Scan record #${scanId} removed from audit ledger`);
        } catch {
            toast.error("Failed to delete record");
        }
    };

    useEffect(() => {
        fetchHistory();
    }, [fetchHistory]);

    // Map backend resultLabel to display status
    const mapResult = (label: string): "Real" | "AI Generated" | "Suspicious" => {
        const lower = label?.toLowerCase() || "";
        if (lower === "fake") return "AI Generated";
        if (lower === "real") return "Real";
        return "Suspicious";
    };

    // Filter, search & sort
    const processedHistory = useMemo(() => {
        const filtered = history.filter((item) => {
            const displayResult = mapResult(item.resultLabel);
            const matchesFilter = filter === "All" || displayResult === filter;
            const query = searchQuery.trim().toLowerCase();
            const matchesSearch =
                query === "" ||
                item.scanId.toString().includes(query) ||
                (item.imagePath && item.imagePath.toLowerCase().includes(query)) ||
                (item.analysisMetadata && item.analysisMetadata.toLowerCase().includes(query));
            return matchesFilter && matchesSearch;
        });

        return filtered.sort((a, b) => {
            if (sortBy === "newest") {
                return new Date(b.scanTimestamp || 0).getTime() - new Date(a.scanTimestamp || 0).getTime();
            }
            if (sortBy === "oldest") {
                return new Date(a.scanTimestamp || 0).getTime() - new Date(b.scanTimestamp || 0).getTime();
            }
            if (sortBy === "highest") {
                return (b.confidenceScore || 0) - (a.confidenceScore || 0);
            }
            if (sortBy === "lowest") {
                return (a.confidenceScore || 0) - (b.confidenceScore || 0);
            }
            return 0;
        });
    }, [history, filter, searchQuery, sortBy]);

    // Reset pagination when query or filter changes
    useEffect(() => {
        setCurrentPage(1);
    }, [filter, searchQuery, sortBy]);

    const totalPages = Math.max(1, Math.ceil(processedHistory.length / itemsPerPage));
    const paginatedItems = useMemo(() => {
        const start = (currentPage - 1) * itemsPerPage;
        return processedHistory.slice(start, start + itemsPerPage);
    }, [processedHistory, currentPage, itemsPerPage]);

    const counts = {
        All: history.length,
        "AI Generated": history.filter(h => mapResult(h.resultLabel) === "AI Generated").length,
        Real: history.filter(h => mapResult(h.resultLabel) === "Real").length,
        Suspicious: history.filter(h => mapResult(h.resultLabel) === "Suspicious").length,
    };

    const handleExportCsv = () => {
        if (processedHistory.length === 0) {
            toast.error("No records available to export");
            return;
        }

        const headers = ["Scan ID", "File Path", "Verdict", "Confidence Score", "Timestamp"];
        const rows = processedHistory.map((item) => [
            item.scanId,
            `"${(item.imagePath || '').replace(/"/g, '""')}"`,
            mapResult(item.resultLabel),
            item.confidenceScore ?? 0,
            item.scanTimestamp || "",
        ]);

        const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `ai-verify-snap-audit-ledger-${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        toast.success(`Exported ${processedHistory.length} audit records as CSV`);
    };

    const handleExportJson = () => {
        if (processedHistory.length === 0) {
            toast.error("No records available to export");
            return;
        }

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(processedHistory, null, 2));
        const link = document.createElement("a");
        link.setAttribute("href", dataStr);
        link.setAttribute("download", `ai-verify-snap-audit-ledger-${Date.now()}.json`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        toast.success(`Exported ${processedHistory.length} audit records as JSON`);
    };

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 sm:p-6 md:p-8 max-w-7xl mx-auto space-y-8">
            {/* Page Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-border/80 pb-6">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <Shield className="w-5 h-5 text-primary" />
                        <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-foreground font-sans">
                            Scan History & Forensic Ledger
                        </h1>
                    </div>
                    <p className="text-muted-foreground text-xs sm:text-sm">
                        Tamper-evident audit log and verification certificates for all analyzed media frames.
                    </p>
                </div>

                <div className="flex flex-wrap items-center gap-2.5 w-full md:w-auto">
                    <button
                        type="button"
                        onClick={handleExportCsv}
                        disabled={processedHistory.length === 0}
                        className="px-3.5 py-2 rounded-xl bg-card border border-border hover:bg-muted text-foreground text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 shadow-sm focus-ring"
                    >
                        <FileSpreadsheet className="w-3.5 h-3.5" />
                        <span>Export CSV</span>
                    </button>
                    <button
                        type="button"
                        onClick={handleExportJson}
                        disabled={processedHistory.length === 0}
                        className="px-3.5 py-2 rounded-xl bg-card border border-border hover:bg-muted text-foreground text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 shadow-sm focus-ring"
                    >
                        <FileCode className="w-3.5 h-3.5" />
                        <span>Export JSON</span>
                    </button>
                    <button
                        type="button"
                        onClick={fetchHistory}
                        disabled={loading}
                        className="px-3.5 py-2 rounded-xl bg-card border border-border hover:bg-muted text-foreground text-xs font-semibold transition-colors flex items-center gap-1.5 shadow-sm focus-ring"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                        <span>Sync</span>
                    </button>
                </div>
            </div>

            {/* Filter, Search & Sort Bar */}
            <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
                {/* Filter Pills */}
                <div className="flex flex-wrap items-center gap-1.5">
                    {(["All", "AI Generated", "Real", "Suspicious"] as const).map((f) => {
                        const isActive = filter === f;
                        return (
                            <button
                                key={f}
                                type="button"
                                onClick={() => setFilter(f)}
                                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 flex items-center gap-1.5 ${
                                    isActive
                                        ? "bg-foreground text-background shadow-sm"
                                        : "bg-card text-muted-foreground hover:bg-muted hover:text-foreground border border-border"
                                }`}
                            >
                                <span>{f}</span>
                                <span
                                    className={`text-[10px] px-1.5 py-0.2 rounded-md font-mono ${
                                        isActive
                                            ? "bg-background/20 text-background"
                                            : "bg-muted text-muted-foreground"
                                    }`}
                                >
                                    {counts[f]}
                                </span>
                            </button>
                        );
                    })}
                </div>

                {/* Search & Sort Controls */}
                <div className="flex flex-col sm:flex-row items-center gap-2.5">
                    {/* Search Input */}
                    <div className="relative w-full sm:w-64">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
                        <input
                            type="text"
                            placeholder="Filter by ID or file..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-8 pr-8 py-1.5 rounded-lg border border-border bg-card text-foreground text-xs focus:outline-none focus:ring-1 focus:ring-foreground transition-all placeholder:text-muted-foreground"
                        />
                        {searchQuery && (
                            <button
                                type="button"
                                onClick={() => setSearchQuery("")}
                                aria-label="Clear search"
                                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                            >
                                <X className="w-3.5 h-3.5" />
                            </button>
                        )}
                    </div>

                    {/* Sort Selector */}
                    <div className="flex items-center gap-1.5 bg-card border border-border px-3 py-1.5 rounded-lg text-xs text-muted-foreground w-full sm:w-auto justify-between sm:justify-start">
                        <ArrowUpDown className="w-3.5 h-3.5 shrink-0" />
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as SortOption)}
                            className="bg-transparent text-foreground text-xs font-semibold focus:outline-none cursor-pointer"
                        >
                            <option value="newest">Newest First</option>
                            <option value="oldest">Oldest First</option>
                            <option value="highest">Highest Score</option>
                            <option value="lowest">Lowest Score</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Loading State */}
            {loading && (
                <div className="flex flex-col items-center justify-center py-24 space-y-4 bg-card border border-border/80 rounded-2xl shadow-sm">
                    <Loader2 className="w-8 h-8 text-primary animate-spin" />
                    <p className="text-xs sm:text-sm font-medium text-muted-foreground">Retrieving cryptographic audit logs...</p>
                </div>
            )}

            {/* Error State */}
            {error && !loading && (
                <div className="p-8 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-center max-w-xl mx-auto space-y-3">
                    <p className="font-semibold text-sm">{error}</p>
                    <button
                        onClick={fetchHistory}
                        className="px-5 py-2 bg-destructive text-destructive-foreground text-xs font-semibold rounded-xl hover:bg-destructive/90 transition-colors"
                    >
                        Retry Connection
                    </button>
                </div>
            )}

            {/* Content Display */}
            {!loading && !error && (
                <div className="space-y-4">
                    {processedHistory.length === 0 ? (
                        <div className="bg-card border border-border/80 rounded-2xl p-12 text-center max-w-lg mx-auto space-y-4 shadow-sm">
                            <div className="w-16 h-16 rounded-2xl bg-muted/60 flex items-center justify-center mx-auto text-muted-foreground border border-border/40">
                                <HardDrive className="w-8 h-8" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">No records matching query</h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                {history.length === 0
                                    ? "No media analyses have been recorded yet in this system. Drop an image to perform your first verification."
                                    : "No scan matches your selected criteria or search term."}
                            </p>
                            {history.length === 0 ? (
                                <Link
                                    href="/detect"
                                    className="inline-flex items-center gap-2 px-6 py-2.5 bg-foreground text-background rounded-xl text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-colors shadow-sm"
                                >
                                    <Sparkles className="w-4 h-4" /> Start First Scan
                                </Link>
                            ) : (
                                <button
                                    onClick={() => { setFilter("All"); setSearchQuery(""); }}
                                    className="px-4 py-2 bg-muted text-foreground rounded-xl text-xs font-semibold hover:bg-muted/80 transition-colors border border-border/60"
                                >
                                    Reset Filters
                                </button>
                            )}
                        </div>
                    ) : (
                        <>
                            {/* Desktop Table View */}
                            <div className="hidden md:block bg-card border border-border/80 rounded-2xl overflow-hidden shadow-sm">
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs text-left">
                                        <thead className="bg-muted/40 text-muted-foreground font-bold uppercase tracking-wider text-[11px] border-b border-border/80">
                                            <tr>
                                                <th scope="col" className="px-6 py-4">Scan Reference</th>
                                                <th scope="col" className="px-6 py-4">Analyzed Media</th>
                                                <th scope="col" className="px-6 py-4">Forensic Verdict</th>
                                                <th scope="col" className="px-6 py-4">Certainty Metric</th>
                                                <th scope="col" className="px-6 py-4">Timestamp</th>
                                                <th scope="col" className="px-6 py-4 text-right">Certificate</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-border/60">
                                            {paginatedItems.map((row) => {
                                                const displayResult = mapResult(row.resultLabel);
                                                const conf = row.confidenceScore ?? 0;
                                                const fileName = row.imagePath ? row.imagePath.replace(/^.*[\\/]/, '') : "unnamed_image.png";

                                                return (
                                                    <tr key={row.scanId} className="hover:bg-muted/30 transition-colors group">
                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                            <span className="font-mono text-xs font-bold text-muted-foreground group-hover:text-foreground transition-colors">
                                                                #{row.scanId}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                            <div className="flex items-center gap-2.5">
                                                                <span className="text-xs font-semibold text-foreground truncate max-w-[200px]" title={row.imagePath}>
                                                                    {fileName}
                                                                </span>
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                            <ResultBadge status={displayResult} size="sm" />
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                            <div className="flex items-center gap-2.5">
                                                                <div className="w-20 h-2 bg-muted rounded-full overflow-hidden border border-border/40">
                                                                    <div
                                                                        className={`h-full rounded-full ${
                                                                            displayResult === "AI Generated"
                                                                                ? "bg-destructive"
                                                                                : displayResult === "Real"
                                                                                    ? "bg-green-600"
                                                                                    : "bg-yellow-500"
                                                                        }`}
                                                                        style={{ width: `${Math.min(100, Math.max(0, conf))}%` }}
                                                                    />
                                                                </div>
                                                                <span className="font-mono text-xs font-bold text-foreground">
                                                                    {conf.toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-muted-foreground text-xs font-mono">
                                                            {row.scanTimestamp ? new Date(row.scanTimestamp).toLocaleString() : "—"}
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-right">
                                                            <div className="flex items-center justify-end gap-2">
                                                                <Link
                                                                    href={`/report/${row.scanId}`}
                                                                    className="inline-flex items-center gap-1.5 text-xs font-bold text-primary hover:underline px-3 py-1.5 rounded-lg hover:bg-muted transition-colors"
                                                                >
                                                                    <span>Certificate</span>
                                                                    <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                                                                </Link>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => handleDeleteItem(row.scanId)}
                                                                    className="p-1.5 text-muted-foreground hover:text-rose-500 rounded-lg hover:bg-muted transition-colors"
                                                                    title="Delete record from ledger"
                                                                >
                                                                    <Trash2 className="w-3.5 h-3.5" />
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Pagination Controls */}
                                <div className="p-4 bg-muted/20 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
                                    <span>
                                        Showing {(currentPage - 1) * itemsPerPage + 1} to {Math.min(currentPage * itemsPerPage, processedHistory.length)} of {processedHistory.length} records
                                    </span>
                                    <div className="flex items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                                            disabled={currentPage === 1}
                                            className="p-1.5 rounded-lg bg-card border border-border text-foreground hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
                                            aria-label="Previous page"
                                        >
                                            <ChevronLeft className="w-4 h-4" />
                                        </button>
                                        <span className="font-mono font-bold text-foreground px-2">
                                            {currentPage} / {totalPages}
                                        </span>
                                        <button
                                            type="button"
                                            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                                            disabled={currentPage === totalPages}
                                            className="p-1.5 rounded-lg bg-card border border-border text-foreground hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
                                            aria-label="Next page"
                                        >
                                            <ChevronRight className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Mobile Card View */}
                            <div className="grid grid-cols-1 gap-3 md:hidden">
                                {paginatedItems.map((row) => {
                                    const displayResult = mapResult(row.resultLabel);
                                    const conf = row.confidenceScore ?? 0;
                                    const fileName = row.imagePath ? row.imagePath.replace(/^.*[\\/]/, '') : "unnamed_image.png";

                                    return (
                                        <div
                                            key={row.scanId}
                                            className="bg-card border border-border rounded-2xl p-4 space-y-3 shadow-sm"
                                        >
                                            <div className="flex items-center justify-between">
                                                <span className="font-mono text-xs font-bold text-muted-foreground">
                                                    #{row.scanId}
                                                </span>
                                                <ResultBadge status={displayResult} size="sm" />
                                            </div>

                                            <p className="font-semibold text-sm text-foreground truncate" title={fileName}>
                                                {fileName}
                                            </p>

                                            <div className="flex items-center justify-between text-xs text-muted-foreground pt-1 border-t border-border/50">
                                                <div className="flex items-center gap-1.5 font-mono">
                                                    <Calendar className="w-3.5 h-3.5" />
                                                    <span>{row.scanTimestamp ? new Date(row.scanTimestamp).toLocaleDateString() : "—"}</span>
                                                </div>
                                                <div className="font-mono font-bold text-foreground">
                                                    {conf.toFixed(1)}% Score
                                                </div>
                                            </div>

                                            <Link
                                                href={`/report/${row.scanId}`}
                                                className="w-full mt-2 py-2.5 bg-muted/80 hover:bg-muted text-foreground text-xs font-bold rounded-xl flex items-center justify-center gap-1.5 transition-colors"
                                            >
                                                <span>View Forensic Certificate</span>
                                                <ArrowRight className="w-3.5 h-3.5" />
                                            </Link>
                                        </div>
                                    );
                                })}

                                {/* Mobile Pagination */}
                                {totalPages > 1 && (
                                    <div className="flex items-center justify-between p-3 bg-card border border-border rounded-xl text-xs text-muted-foreground">
                                        <button
                                            type="button"
                                            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                                            disabled={currentPage === 1}
                                            className="px-3 py-1.5 bg-muted rounded-lg text-foreground font-semibold disabled:opacity-40"
                                        >
                                            Previous
                                        </button>
                                        <span className="font-mono">
                                            Page {currentPage} of {totalPages}
                                        </span>
                                        <button
                                            type="button"
                                            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                                            disabled={currentPage === totalPages}
                                            className="px-3 py-1.5 bg-muted rounded-lg text-foreground font-semibold disabled:opacity-40"
                                        >
                                            Next
                                        </button>
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
