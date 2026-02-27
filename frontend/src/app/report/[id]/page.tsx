"use client";

import { Download, FileSignature, ShieldCheck, QrCode } from "lucide-react";
import Image from "next/image";
import { ResultBadge } from "@/components/shared/ResultBadge";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";

export default function ReportPage() {
    const { id } = useParams();
    const mockReport = {
        id: id || "REV-4892",
        result: "AI Generated",
        confidence: 98,
        timestamp: new Date().toLocaleString(),
        model: "Multi-modal Ensemble (ELA + CNN)",
        imageThumb: "https://images.unsplash.com/photo-1544365558-35aa4afcf11f?w=400&q=80",
        authenticityHash: "7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b",
    };

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 md:p-8 flex items-center justify-center bg-muted/20">
            <div className="max-w-3xl w-full space-y-6">
                <div className="flex items-center justify-between">
                    <h1 className="text-2xl font-bold tracking-tight">Forensic Report</h1>
                    <button className="flex items-center gap-2 px-5 py-2.5 bg-primary text-primary-foreground font-semibold rounded-xl hover:bg-primary/90 transition-all shadow-sm">
                        <Download className="w-4 h-4" /> Download PDF
                    </button>
                </div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-card border-2 shadow-xl p-8 md:p-12 rounded-[2rem] space-y-10 relative overflow-hidden"
                >
                    {/* Watermark */}
                    <ShieldCheck className="absolute -bottom-20 -right-20 w-96 h-96 text-primary/[0.03] z-0 pointer-events-none" />

                    {/* Header */}
                    <div className="flex flex-col md:flex-row justify-between items-start border-b pb-8 relative z-10 gap-6">
                        <div className="flex items-center gap-3 mb-2">
                            <ShieldCheck className="w-10 h-10 text-primary" />
                            <div>
                                <h2 className="text-2xl font-black uppercase tracking-widest text-foreground">AIVerifySnap</h2>
                                <p className="text-xs text-muted-foreground uppercase tracking-widest">Official Verification Certificate</p>
                            </div>
                        </div>
                        <div className="text-right flex flex-col items-end">
                            <span className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Report ID</span>
                            <span className="font-mono text-lg font-bold text-foreground bg-muted px-3 py-1 rounded-lg mt-1">{mockReport.id}</span>
                        </div>
                    </div>

                    {/* Content section */}
                    <div className="grid md:grid-cols-2 gap-10 relative z-10">
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest mb-3">Analyzed Media</h3>
                                <div className="aspect-square relative rounded-2xl overflow-hidden border-2 bg-muted p-2 shadow-inner">
                                    <Image src={mockReport.imageThumb} alt="Subject media" fill className="object-cover rounded-xl" />
                                </div>
                            </div>
                        </div>

                        <div className="space-y-8 flex flex-col justify-center">
                            <div className="p-6 bg-muted/30 border rounded-2xl shadow-sm text-center">
                                <h3 className="text-sm font-semibold text-muted-foreground uppercase mb-2">Final Verdict</h3>
                                <div className="flex justify-center mb-2">
                                    <ResultBadge status={mockReport.result as "Real" | "AI Generated" | "Suspicious"} />
                                </div>
                                <div className="text-4xl font-black mt-4 font-mono">{mockReport.confidence}%</div>
                                <p className="text-muted-foreground text-sm mt-1">Confidence Score</p>
                            </div>

                            <div className="space-y-4">
                                <div className="flex justify-between border-b pb-2">
                                    <span className="text-sm text-muted-foreground">Detection Engine</span>
                                    <span className="text-sm font-semibold">{mockReport.model}</span>
                                </div>
                                <div className="flex justify-between border-b pb-2">
                                    <span className="text-sm text-muted-foreground">Timestamp</span>
                                    <span className="text-sm font-semibold">{mockReport.timestamp}</span>
                                </div>
                                <div className="flex flex-col gap-1 border-b pb-2">
                                    <span className="text-sm text-muted-foreground">Authenticity Hash (SHA-256)</span>
                                    <span className="text-xs font-mono font-medium truncate text-foreground/80">{mockReport.authenticityHash}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Footer of the Report */}
                    <div className="flex items-end justify-between pt-8 border-t relative z-10">
                        <div className="flex items-center gap-3">
                            <QrCode className="w-16 h-16 p-2 bg-background border rounded-lg shadow-sm" />
                            <div className="text-xs text-muted-foreground max-w-[200px]">
                                Scan QR code to verify this certificate&apos;s authenticity online.
                            </div>
                        </div>

                        <div className="text-right">
                            <FileSignature className="w-12 h-12 text-muted-foreground/30 inline-block mb-2" />
                            <div className="text-xs text-muted-foreground uppercase tracking-widest font-semibold border-t-2 pt-1">
                                Authorized Signature
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
