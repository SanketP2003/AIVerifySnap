"use client";

import Link from "next/link";
import {
    UploadCloud,
    Cpu,
    FileCheck,
    ArrowRight,
    CheckCircle2,
    AlertCircle,
    PlusCircle
} from "lucide-react";

export default function HowItWorksPage() {
    return (
        <div className="flex flex-col items-center justify-center bg-background min-h-screen relative overflow-hidden">
            {/* Ambient Lighting */}
            <div className="subtle-blue-light top-0 left-[-200px]" aria-hidden="true" />
            <div className="subtle-orange-light top-[400px] right-[-200px]" aria-hidden="true" />

            {/* Header Hero */}
            <section className="relative w-full px-4 sm:px-6 lg:px-8 pt-16 pb-20 sm:pt-24 sm:pb-28 text-center max-w-5xl mx-auto space-y-6">
                <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-1.5 text-xs sm:text-sm font-semibold text-foreground backdrop-blur-md shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-[#e57a40] animate-pulse" />
                    Forensic Methodology &middot; Scientific Workflow
                </div>

                <h1 className="text-4xl sm:text-5xl md:text-7xl font-serif text-foreground leading-[1.1] tracking-tight">
                    How AIVerifySnap <br />
                    <span className="text-foreground/85 font-serif">Verifies Digital Media.</span>
                </h1>

                <p className="text-base sm:text-xl text-muted-foreground max-w-2xl mx-auto font-normal leading-relaxed">
                    A transparent, three-step inspection architecture designed to separate generative model patterns from authentic optical captures.
                </p>

                <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 pt-4">
                    <Link
                        href="/detect"
                        className="w-full sm:w-auto px-7 py-3.5 bg-foreground text-background font-semibold rounded-xl hover:bg-foreground/90 transition-all text-sm shadow-sm flex items-center justify-center gap-2"
                    >
                        <PlusCircle className="w-4 h-4" />
                        <span>Analyze an Image</span>
                    </Link>
                    <Link
                        href="/dashboard"
                        className="w-full sm:w-auto px-7 py-3.5 bg-card border border-border text-foreground font-semibold rounded-xl hover:bg-muted transition-all text-sm shadow-sm flex items-center justify-center gap-2"
                    >
                        <span>Open Dashboard</span>
                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                    </Link>
                </div>
            </section>

            {/* Step-by-Step Breakdown */}
            <section className="w-full py-16 sm:py-24 border-t border-border/60 bg-muted/20">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-16">
                    <div className="text-center max-w-xl mx-auto space-y-2">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Sequential Pipeline
                        </span>
                        <h2 className="text-3xl sm:text-4xl font-serif text-foreground">
                            The Three-Stage Architecture
                        </h2>
                    </div>

                    <div className="space-y-8">
                        {/* Step 1 */}
                        <div className="p-8 sm:p-10 rounded-2xl bg-card border border-border/80 shadow-sm grid md:grid-cols-12 gap-8 items-center">
                            <div className="md:col-span-4 space-y-3">
                                <div className="flex items-center gap-3">
                                    <span className="font-mono text-3xl font-black text-primary/30">01</span>
                                    <div className="p-2.5 rounded-xl bg-primary/10 text-primary border border-primary/20">
                                        <UploadCloud className="w-5 h-5" />
                                    </div>
                                </div>
                                <h3 className="text-xl font-bold text-foreground font-sans">
                                    Target Ingest & Fingerprinting
                                </h3>
                                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                    Submit your target photograph in JPEG, PNG, or WebP format.
                                </p>
                            </div>
                            <div className="md:col-span-8 p-6 rounded-xl bg-muted/30 border border-border/40 space-y-3 text-xs text-muted-foreground leading-relaxed">
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Deterministic SHA-256 Hashing:</strong> An immutable digital fingerprint is generated prior to transmission, creating a tamper-evident audit record.
                                    </span>
                                </div>
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Compression-Preserving Scaling:</strong> High-resolution inputs are pre-scaled in your browser to 512px boundaries to standardize neural ingestion while preserving localized compression artifacts.
                                    </span>
                                </div>
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Direct Stream Ingestion:</strong> Files stream directly to the backend analysis controller via multipart form requests.
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Step 2 */}
                        <div className="p-8 sm:p-10 rounded-2xl bg-card border border-border/80 shadow-sm grid md:grid-cols-12 gap-8 items-center">
                            <div className="md:col-span-4 space-y-3">
                                <div className="flex items-center gap-3">
                                    <span className="font-mono text-3xl font-black text-primary/30">02</span>
                                    <div className="p-2.5 rounded-xl bg-primary/10 text-primary border border-primary/20">
                                        <Cpu className="w-5 h-5" />
                                    </div>
                                </div>
                                <h3 className="text-xl font-bold text-foreground font-sans">
                                    Dual-Stream Forensic Analysis
                                </h3>
                                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                    The image enters concurrent neural and mathematical evaluation pipelines.
                                </p>
                            </div>
                            <div className="md:col-span-8 p-6 rounded-xl bg-muted/30 border border-border/40 space-y-3 text-xs text-muted-foreground leading-relaxed">
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>SigLIP Vision Transformer:</strong> Examines attention embeddings and high-frequency spectral noise patterns characteristic of latent diffusion models and generative transformers.
                                    </span>
                                </div>
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Error Level Analysis (ELA):</strong> Intentionally compresses the image at 90% quality and calculates the absolute pixel difference in the DCT frequency domain.
                                    </span>
                                </div>
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Attention Mapping:</strong> Extracts spatial gradient activations indicating which image regions contributed most strongly to the model prediction.
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Step 3 */}
                        <div className="p-8 sm:p-10 rounded-2xl bg-card border border-border/80 shadow-sm grid md:grid-cols-12 gap-8 items-center">
                            <div className="md:col-span-4 space-y-3">
                                <div className="flex items-center gap-3">
                                    <span className="font-mono text-3xl font-black text-primary/30">03</span>
                                    <div className="p-2.5 rounded-xl bg-primary/10 text-primary border border-primary/20">
                                        <FileCheck className="w-5 h-5" />
                                    </div>
                                </div>
                                <h3 className="text-xl font-bold text-foreground font-sans">
                                    Verdict & Verifiable Certificate
                                </h3>
                                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                    Review the synthesized forensic results and export verifiable records.
                                </p>
                            </div>
                            <div className="md:col-span-8 p-6 rounded-xl bg-muted/30 border border-border/40 space-y-3 text-xs text-muted-foreground leading-relaxed">
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Clear Verdict Classification:</strong> Instant classification badge (&ldquo;AI Generated&rdquo; vs &ldquo;Authentic&rdquo;) accompanied by calibrated confidence percentages.
                                    </span>
                                </div>
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Interactive Comparison View:</strong> Slide between the original photograph, Error Level Analysis residual delta, and neural attention maps.
                                    </span>
                                </div>
                                <div className="flex items-start gap-2.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                                    <span>
                                        <strong>Exportable Certificate:</strong> Every scan is stamped into the ledger with a shareable certificate URL and downloadable JSON technical dossier.
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Scientific Disclaimer */}
                    <div className="p-6 rounded-2xl bg-card border border-border/80 text-center space-y-3 max-w-3xl mx-auto shadow-sm">
                        <div className="flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
                            <AlertCircle className="w-4 h-4 text-[#e57a40]" />
                            <span>Scientific Decision Support Notice</span>
                        </div>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                            AIVerifySnap provides probabilistic model classifications and deterministic compression error metrics to assist analysts, journalists, and researchers. No automated system can provide an infallible guarantee of an image&apos;s real-world authenticity.
                        </p>
                    </div>
                </div>
            </section>

            {/* Bottom CTA */}
            <section className="w-full py-20 bg-muted/30 border-t border-border/60 text-center">
                <div className="max-w-3xl mx-auto px-4 sm:px-6 space-y-6">
                    <h2 className="text-3xl sm:text-4xl font-serif text-foreground">
                        Ready to Examine Your Media?
                    </h2>
                    <p className="text-sm sm:text-base text-muted-foreground">
                        Test your images with our verified dual-stream forensic pipeline now.
                    </p>
                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
                        <Link
                            href="/detect"
                            className="w-full sm:w-auto px-8 py-3.5 bg-foreground text-background font-semibold rounded-full hover:bg-foreground/90 transition-all text-sm shadow-md"
                        >
                            Start an Analysis
                        </Link>
                        <Link
                            href="/dashboard"
                            className="w-full sm:w-auto px-8 py-3.5 bg-card border border-border text-foreground font-semibold rounded-full hover:bg-muted transition-all text-sm shadow-sm"
                        >
                            Open Dashboard Workspace &rarr;
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}
