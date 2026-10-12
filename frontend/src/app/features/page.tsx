"use client";

import Link from "next/link";
import {
    Cpu,
    Layers,
    Sparkles,
    ShieldCheck,
    History,
    Fingerprint,
    ArrowRight,
    CheckCircle2,
    PlusCircle
} from "lucide-react";

export default function FeaturesPage() {
    return (
        <div className="flex flex-col items-center justify-center bg-background min-h-screen relative overflow-hidden">
            {/* Ambient Lighting */}
            <div className="subtle-blue-light top-0 left-[-200px]" aria-hidden="true" />
            <div className="subtle-orange-light top-[400px] right-[-200px]" aria-hidden="true" />

            {/* Header Hero */}
            <section className="relative w-full px-4 sm:px-6 lg:px-8 pt-16 pb-20 sm:pt-24 sm:pb-28 text-center max-w-5xl mx-auto space-y-6">
                <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-1.5 text-xs sm:text-sm font-semibold text-foreground backdrop-blur-md shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-[#e57a40] animate-pulse" />
                    Verified Forensic Capabilities &middot; v2.4 Architecture
                </div>

                <h1 className="text-4xl sm:text-5xl md:text-7xl font-serif text-foreground leading-[1.1] tracking-tight">
                    Engineered for Rigorous <br />
                    <span className="text-foreground/85 font-serif">Media Verification.</span>
                </h1>

                <p className="text-base sm:text-xl text-muted-foreground max-w-2xl mx-auto font-normal leading-relaxed">
                    Explore the complete suite of deterministic forensic signals and vision transformer classifiers implemented in AIVerifySnap.
                </p>

                <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 pt-4">
                    <Link
                        href="/detect"
                        className="w-full sm:w-auto px-8 py-4 bg-foreground text-background font-semibold rounded-full hover:bg-foreground/90 transition-all text-base shadow-md flex items-center justify-center gap-2"
                    >
                        <PlusCircle className="w-5 h-5" />
                        <span>Start Analyzing Images</span>
                    </Link>
                    <Link
                        href="/dashboard"
                        className="w-full sm:w-auto px-8 py-4 bg-card border border-border text-foreground font-semibold rounded-full hover:bg-muted transition-all text-base shadow-sm flex items-center justify-center gap-2"
                    >
                        <span>Open Dashboard</span>
                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                    </Link>
                </div>
            </section>

            {/* In-Depth Feature Grid */}
            <section className="w-full py-16 sm:py-24 border-t border-border/60 bg-muted/20">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 space-y-16">
                    <div className="text-center max-w-2xl mx-auto space-y-2">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Technical Inspection Stack
                        </span>
                        <h2 className="text-3xl sm:text-4xl font-serif text-foreground">
                            Multi-Layer Forensic Signals
                        </h2>
                        <p className="text-sm text-muted-foreground">
                            Combining transformer pattern classification with spatial compression physics.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {/* 1 */}
                        <div className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 hover:border-foreground/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                <Cpu className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">
                                SigLIP Vision Transformer
                            </h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Evaluates high-frequency spectral artifacts and latent diffusion traces characteristic of generative models including Midjourney, Stable Diffusion, and DALL-E.
                            </p>
                            <div className="text-[11px] font-mono font-bold text-primary bg-primary/10 px-2.5 py-1 rounded-md w-fit">
                                Probabilistic Embedding Model
                            </div>
                        </div>

                        {/* 2 */}
                        <div className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 hover:border-foreground/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                <Layers className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">
                                Error Level Analysis (ELA)
                            </h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Computes the discrete cosine transform (DCT) residual delta after intentional 90% JPEG recompression. Localized brightness anomalies reveal modified image sections.
                            </p>
                            <div className="text-[11px] font-mono font-bold text-green-600 bg-green-500/10 px-2.5 py-1 rounded-md w-fit">
                                Deterministic Residual Delta
                            </div>
                        </div>

                        {/* 3 */}
                        <div className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 hover:border-foreground/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                <Sparkles className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">
                                Attention Heatmap Mapping
                            </h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Renders spatial gradient activation overlays, exposing exactly which features (eyes, texture boundaries, background artifacts) influenced the neural classification.
                            </p>
                            <div className="text-[11px] font-mono font-bold text-muted-foreground bg-muted px-2.5 py-1 rounded-md w-fit">
                                Spatial Heatmap Inspection
                            </div>
                        </div>

                        {/* 4 */}
                        <div className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 hover:border-foreground/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                <ShieldCheck className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">
                                Cryptographic SHA-256 Ledger
                            </h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Calculates deterministic SHA-256 hashes for analyzed frames, establishing verifiable proof of custody and tamper-evident audit records.
                            </p>
                            <div className="text-[11px] font-mono font-bold text-muted-foreground bg-muted px-2.5 py-1 rounded-md w-fit">
                                Cryptographic Fingerprinting
                            </div>
                        </div>

                        {/* 5 */}
                        <div className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 hover:border-foreground/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                <History className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">
                                Searchable Audit Ledger
                            </h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Full scan ledger with multi-column filtering, search by hash or filename, chronological sorting, and instantaneous CSV & JSON technical dossier exports.
                            </p>
                            <div className="text-[11px] font-mono font-bold text-muted-foreground bg-muted px-2.5 py-1 rounded-md w-fit">
                                Auditability & Export
                            </div>
                        </div>

                        {/* 6 */}
                        <div className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 hover:border-foreground/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                <Fingerprint className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-foreground">
                                Digital Likeness Protection
                            </h3>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Register reference biometric facial profiles to detect unauthorized synthetic clones and assemble structured intellectual property infringement reports.
                            </p>
                            <div className="text-[11px] font-mono font-bold text-muted-foreground bg-muted px-2.5 py-1 rounded-md w-fit">
                                Likeness Defense Workflow
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Methodology Comparison Section */}
            <section className="w-full py-20 sm:py-28 relative">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-12">
                    <div className="text-center max-w-2xl mx-auto space-y-2">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Methodology Transparency
                        </span>
                        <h2 className="text-3xl sm:text-4xl font-serif text-foreground">
                            Comparing Forensic Evidence Signals
                        </h2>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="p-8 rounded-3xl bg-card border border-border shadow-sm space-y-4">
                            <div className="flex items-center gap-2">
                                <Cpu className="w-5 h-5 text-primary" />
                                <h3 className="font-bold text-foreground text-lg">Machine Learning Inference</h3>
                            </div>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Pre-trained deep neural architectures identify abstract high-dimensional patterns that distinguish artificial generation from genuine optical capture.
                            </p>
                            <ul className="text-xs text-muted-foreground space-y-2 pt-2">
                                <li className="flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                    <span>Trained on extensive synthetic and real image corpora</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                    <span>High accuracy on state-of-the-art diffusion outputs</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                    <span>Provides calibrated certainty score (%)</span>
                                </li>
                            </ul>
                        </div>

                        <div className="p-8 rounded-3xl bg-card border border-border shadow-sm space-y-4">
                            <div className="flex items-center gap-2">
                                <Layers className="w-5 h-5 text-primary" />
                                <h3 className="font-bold text-foreground text-lg">Signal & Compression Forensics</h3>
                            </div>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Deterministic signal mathematics examine the physical lossy compression history of image blocks, revealing spliced elements and divergent quantization tables.
                            </p>
                            <ul className="text-xs text-muted-foreground space-y-2 pt-2">
                                <li className="flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                    <span>Direct pixel-level residual calculation</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                    <span>Independent of machine learning bias</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                                    <span>Visual heatmap highlighting for human inspection</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            {/* Bottom CTA */}
            <section className="w-full py-20 bg-muted/30 border-t border-border/60 text-center">
                <div className="max-w-3xl mx-auto px-4 sm:px-6 space-y-6">
                    <h2 className="text-3xl sm:text-4xl font-serif text-foreground">
                        Ready to Test Your Media?
                    </h2>
                    <p className="text-sm sm:text-base text-muted-foreground">
                        Upload any image to experience the multi-tiered detection pipeline firsthand.
                    </p>
                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
                        <Link
                            href="/detect"
                            className="w-full sm:w-auto px-8 py-3.5 bg-foreground text-background font-semibold rounded-full hover:bg-foreground/90 transition-all text-sm shadow-md"
                        >
                            Analyze an Image Now
                        </Link>
                        <Link
                            href="/how-it-works"
                            className="w-full sm:w-auto px-8 py-3.5 bg-card border border-border text-foreground font-semibold rounded-full hover:bg-muted transition-all text-sm shadow-sm"
                        >
                            Review Verification Workflow &rarr;
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}
