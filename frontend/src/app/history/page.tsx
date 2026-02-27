"use client";

import { Search, Filter, ArrowRight } from "lucide-react";
import Image from "next/image";
import { ResultBadge } from "@/components/shared/ResultBadge";
import Link from "next/link";
import { useState } from "react";

const mockHistory = [
    { id: "REV-4892", thumb: "https://images.unsplash.com/photo-1544365558-35aa4afcf11f?w=100&q=80", result: "AI Generated", conf: 98, date: "2026-02-21 14:32" },
    { id: "REV-4891", thumb: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&q=80", result: "Real", conf: 12, date: "2026-02-20 09:15" },
    { id: "REV-4890", thumb: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=100&q=80", result: "Suspicious", conf: 65, date: "2026-02-19 18:45" },
];

export default function HistoryPage() {
    const [filter, setFilter] = useState("All");

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
                            placeholder="Search by ID..."
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

            <div className="bg-card border rounded-[2rem] overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-muted/50 text-muted-foreground font-semibold uppercase tracking-wider text-xs">
                            <tr>
                                <th className="px-6 py-4">Image</th>
                                <th className="px-6 py-4">Result</th>
                                <th className="px-6 py-4 hidden sm:table-cell">Confidence</th>
                                <th className="px-6 py-4 hidden md:table-cell">Date Scanned</th>
                                <th className="px-6 py-4 text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {mockHistory.map((row) => (
                                <tr key={row.id} className="hover:bg-muted/20 transition-colors group">
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="flex items-center gap-4">
                                            <div className="w-12 h-12 relative rounded-xl overflow-hidden border">
                                                <Image src={row.thumb} alt="thumbnail" fill className="object-cover" />
                                            </div>
                                            <span className="font-mono text-xs text-muted-foreground group-hover:text-foreground transition-colors">{row.id}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <ResultBadge status={row.result as "Real" | "AI Generated" | "Suspicious"} />
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap hidden sm:table-cell">
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                                                <div className={`h-full ${row.conf > 75 ? 'bg-destructive' : row.conf > 40 ? 'bg-yellow-500' : 'bg-green-500'}`} style={{ width: `${row.conf}%` }} />
                                            </div>
                                            <span className="font-medium">{row.conf}%</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-muted-foreground hidden md:table-cell">
                                        {row.date}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right">
                                        <Link href={`/report/${row.id}`} className="inline-flex items-center gap-1 text-primary hover:text-primary/80 font-medium transition-colors">
                                            View Report <ArrowRight className="w-4 h-4" />
                                        </Link>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
