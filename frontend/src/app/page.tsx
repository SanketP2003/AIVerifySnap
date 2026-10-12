"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
    ArrowRight,
    ShieldCheck,
    Cpu,
    Fingerprint,
    History,
    Sparkles,
    Layers,
    ChevronDown,
    Lock,
    PlusCircle
} from "lucide-react";
import Link from "next/link";
import { ResultBadge } from "@/components/shared/ResultBadge";

const containerVariants = {
    hidden: {},
    visible: {
        transition: { staggerChildren: 0.08 },
    },
};

const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] as const } },
};

const faqs = [
    {
        question: "What does AIVerifySnap analyze?",
        answer: "AIVerifySnap inspects digital photographs for signs of synthetic generation from diffusion models (such as Midjourney, DALL-E, Stable Diffusion) and generative vision transformers, as well as digital image manipulation and splicing via localized JPEG compression error analysis."
    },
    {
        question: "How do I upload an image?",
        answer: "Navigate to the New Analysis workspace, then drag and drop any JPEG, PNG, or WebP photograph (up to 10 MB) into the dropzone, or click 'Select File'. The image is prepared and streamed directly to our forensic inference pipeline."
    },
    {
        question: "What does an AI detection result mean?",
        answer: "The verdict indicates the statistical probability that high-frequency frequency features and visual attention embeddings align with known synthetic training distributions. A classification of 'AI Generated' reflects neural generative patterns, whereas 'Authentic' reflects camera sensor noise and natural optical characteristics."
    },
    {
        question: "What is Error Level Analysis (ELA)?",
        answer: "Error Level Analysis intentional re-saves the image at a known JPEG quantization level and computes the pixel-level difference. Areas with distinct compression error deltas indicate potential editing, splicing, or synthetic generation."
    },
    {
        question: "Are the results legally definitive?",
        answer: "No automated AI classifier can provide an absolute guarantee of authenticity. AIVerifySnap is designed as a forensic decision-support platform, providing verifiable signal evidence (ELA mean, standard deviation, Grad-CAM attention) that human examiners can cross-reference."
    },
    {
        question: "Can I download an analysis report?",
        answer: "Yes. Every completed scan generates a verification ledger record with an official verification certificate, cryptographic SHA-256 hash, and exportable JSON technical dossier."
    }
];

export default function PublicLandingPage() {
    const [openFaq, setOpenFaq] = useState<number | null>(null);
    const [previewMode, setPreviewMode] = useState<"original" | "ela" | "heatmap">("ela");

    const toggleFaq = (index: number) => {
        setOpenFaq(openFaq === index ? null : index);
    };

    return (
        <div className="flex flex-col items-center justify-center bg-background min-h-screen relative overflow-hidden">
            {/* Ambient background lights */}
            <div className="subtle-blue-light top-0 left-[-200px]" aria-hidden="true" />
            <div className="subtle-orange-light top-[400px] right-[-200px]" aria-hidden="true" />

            {/* ========================================================= */}
            {/* SECTION B: HERO SECTION */}
            {/* ========================================================= */}
            <section className="relative w-full px-4 sm:px-6 lg:px-8 pt-12 pb-20 sm:pt-20 sm:pb-28 lg:pt-28 lg:pb-36 flex flex-col items-center justify-center overflow-hidden">
                {/* Soft Spotlight */}
                <div className="absolute top-10 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-[#e7af55]/5 rounded-full blur-[120px] -z-10 pointer-events-none" aria-hidden="true" />

                <motion.div
                    className="text-center space-y-6 max-w-[920px] mx-auto z-10"
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                >
                    <motion.div variants={itemVariants}>
                        <div className="mb-6 sm:mb-8">
                            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-1.5 text-xs sm:text-sm font-semibold text-foreground backdrop-blur-md shadow-sm">
                                <span className="w-2 h-2 rounded-full bg-[#e57a40] animate-pulse" />
                                Sovereign Digital Forensics & Media Verification
                            </span>
                        </div>
                        <h1 className="text-4xl sm:text-5xl md:text-7xl lg:text-8xl font-serif text-foreground text-balance leading-[1.08] mb-4 tracking-tight">
                            Know What&apos;s Behind <br />
                            <span className="text-foreground/85 font-serif">the Image.</span>
                        </h1>
                    </motion.div>

                    <motion.p
                        className="text-base sm:text-xl md:text-2xl text-muted-foreground text-balance max-w-2xl mx-auto font-normal leading-relaxed"
                        variants={itemVariants}
                    >
                        Analyze images for signs of AI generation and potential manipulation using the forensic analysis capabilities available in AIVerifySnap.
                    </motion.p>

                    <motion.div
                        className="flex flex-col sm:flex-row items-center justify-center gap-3.5 pt-4"
                        variants={itemVariants}
                    >
                        <Link
                            href="/detect"
                            className="w-full sm:w-auto px-8 py-4 bg-foreground text-background font-semibold rounded-full hover:bg-foreground/90 transition-all duration-200 active:scale-[0.98] text-base text-center focus-ring shadow-md flex items-center justify-center gap-2"
                        >
                            <PlusCircle className="w-5 h-5" />
                            <span>Analyze an Image</span>
                        </Link>
                        <a
                            href="#features"
                            className="w-full sm:w-auto px-8 py-4 bg-card border border-border text-foreground font-semibold rounded-full hover:bg-muted transition-all duration-200 active:scale-[0.98] text-base text-center focus-ring shadow-sm flex items-center justify-center gap-2"
                        >
                            <span>Explore Features</span>
                            <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        </a>
                    </motion.div>
                </motion.div>

                {/* ========================================================= */}
                {/* SECTION C: PRODUCT PREVIEW / INTERACTIVE DEMONSTRATION */}
                {/* ========================================================= */}
                <motion.div
                    className="mt-14 sm:mt-20 max-w-5xl w-full z-10"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.25, duration: 0.7 }}
                >
                    <div className="bg-card border-2 border-border/80 rounded-[2.5rem] p-5 sm:p-8 shadow-2xl relative overflow-hidden">
                        {/* Demonstration Badge Header */}
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-border/60 gap-3 mb-6">
                            <div className="flex items-center gap-2.5">
                                <span className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
                                <span className="font-mono text-xs font-bold uppercase tracking-wider text-muted-foreground">
                                    Forensic Inspection Demonstration
                                </span>
                            </div>
                            <span className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-muted text-muted-foreground border border-border/60">
                                Illustrative Demonstration &middot; Nonfunctional Preview
                            </span>
                        </div>

                        {/* Interactive Preview Canvas */}
                        <div className="grid md:grid-cols-12 gap-6 items-center">
                            {/* Visual Preview Frame (7 cols) */}
                            <div className="md:col-span-7 space-y-3">
                                <div className="flex items-center gap-1.5 bg-muted/60 p-1 rounded-2xl w-fit text-xs font-semibold">
                                    <button
                                        type="button"
                                        onClick={() => setPreviewMode("original")}
                                        className={`px-3 py-1.5 rounded-xl transition-all ${previewMode === "original" ? "bg-background shadow text-foreground" : "text-muted-foreground"}`}
                                    >
                                        Original Frame
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setPreviewMode("ela")}
                                        className={`px-3 py-1.5 rounded-xl transition-all ${previewMode === "ela" ? "bg-background shadow text-foreground" : "text-muted-foreground"}`}
                                    >
                                        ELA Recompression
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setPreviewMode("heatmap")}
                                        className={`px-3 py-1.5 rounded-xl transition-all ${previewMode === "heatmap" ? "bg-background shadow text-foreground" : "text-muted-foreground"}`}
                                    >
                                        Attention Map
                                    </button>
                                </div>

                                <div className="relative aspect-[4/3] rounded-2xl overflow-hidden bg-neutral-900 border border-border flex items-center justify-center">
                                    {/* Mock Subject Face */}
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                        src="https://images.unsplash.com/photo-1544365558-35aa4afcf11f?w=600&q=80"
                                        alt="Demonstration Subject"
                                        className={`w-full h-full object-cover transition-all duration-300 ${previewMode === "ela" ? "filter contrast-200 brightness-50 invert" : previewMode === "heatmap" ? "opacity-60" : ""}`}
                                    />
                                    {previewMode === "heatmap" && (
                                        <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/40 via-yellow-400/40 to-red-500/60 mix-blend-overlay filter blur-xl pointer-events-none" />
                                    )}
                                    <div className="absolute bottom-3 left-3 px-3 py-1 rounded-full bg-background/80 backdrop-blur-md text-[11px] font-mono text-foreground border border-border shadow-sm">
                                        {previewMode === "original" ? "Raw Input: camera_portrait.png" : previewMode === "ela" ? "ELA Residual Delta (Quantization error)" : "SigLIP Transformer Feature Activation"}
                                    </div>
                                </div>
                            </div>

                            {/* Evidence Summary Column (5 cols) */}
                            <div className="md:col-span-5 space-y-4">
                                <div className="p-5 rounded-2xl bg-muted/40 border border-border/60 space-y-3">
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                                            Sample Verification Verdict
                                        </span>
                                        <ResultBadge status="AI Generated" size="sm" />
                                    </div>

                                    <div className="text-3xl font-black font-mono tracking-tight text-destructive">
                                        98.4%
                                    </div>
                                    <p className="text-xs text-muted-foreground">
                                        Confidence metric calculated via SigLIP Vision Transformer embedding classification.
                                    </p>
                                </div>

                                <div className="space-y-2 text-xs">
                                    <div className="flex justify-between p-2.5 rounded-xl bg-card border border-border/50">
                                        <span className="text-muted-foreground">ELA Mean Residual</span>
                                        <span className="font-mono font-bold text-foreground">0.0384</span>
                                    </div>
                                    <div className="flex justify-between p-2.5 rounded-xl bg-card border border-border/50">
                                        <span className="text-muted-foreground">ELA Std Deviation</span>
                                        <span className="font-mono font-bold text-foreground">0.0521</span>
                                    </div>
                                    <div className="flex justify-between p-2.5 rounded-xl bg-card border border-border/50">
                                        <span className="text-muted-foreground">Inference Engine</span>
                                        <span className="font-bold text-foreground">SigLIP ViT Ensemble</span>
                                    </div>
                                </div>

                                <Link
                                    href="/detect"
                                    className="w-full py-3 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all text-center flex items-center justify-center gap-2 shadow-sm"
                                >
                                    <span>Run Real Image Verification</span>
                                    <ArrowRight className="w-4 h-4" />
                                </Link>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </section>

            {/* ========================================================= */}
            {/* SECTION D: FEATURES SECTION */}
            {/* ========================================================= */}
            <section id="features" className="w-full py-20 sm:py-28 border-t border-border/60 bg-muted/20 relative z-10">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 space-y-12">
                    <div className="text-center max-w-2xl mx-auto space-y-3">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Core Forensic Capabilities
                        </span>
                        <h2 className="text-3xl sm:text-4xl md:text-5xl font-serif text-foreground">
                            Multi-Tiered Forensic Architecture
                        </h2>
                        <p className="text-sm sm:text-base text-muted-foreground">
                            Built directly on verified features implemented in the AIVerifySnap codebase.
                        </p>
                    </div>

                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[
                            {
                                icon: Cpu,
                                title: "AI-Generated Image Detection",
                                desc: "Vision Transformer inference classifies high-frequency synthetic artifacts and diffusion noise patterns from generative engines."
                            },
                            {
                                icon: Layers,
                                title: "Error Level Analysis (ELA)",
                                desc: "Recompresses target media at designated quantization levels to reveal localized compression rate differences and edited regions."
                            },
                            {
                                icon: Sparkles,
                                title: "Attention Heatmap Visualization",
                                desc: "Extracts spatial feature activations to demonstrate exactly which image regions contributed to the model classification."
                            },
                            {
                                icon: ShieldCheck,
                                title: "Cryptographic SHA-256 Ledger",
                                desc: "Calculates unique SHA-256 digital fingerprint hashes for analyzed media frames, creating tamper-evident records."
                            },
                            {
                                icon: History,
                                title: "Scan History & Ledger Export",
                                desc: "Maintains a full audit trail of previous analyses with multi-column sorting, pagination, and one-click CSV / JSON export."
                            },
                            {
                                icon: Fingerprint,
                                title: "Likeness Protection Workflow",
                                desc: "Upload a reference biometric portrait to search simulated open web indices and compile infringement evidence packages."
                            }
                        ].map((feat, i) => (
                            <div
                                key={i}
                                className="p-6 sm:p-7 rounded-3xl bg-card border border-border shadow-sm flex flex-col justify-between space-y-4 hover:border-foreground/30 hover:shadow-md transition-all"
                            >
                                <div className="space-y-3">
                                    <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                                        <feat.icon className="w-6 h-6" />
                                    </div>
                                    <h3 className="text-lg font-bold text-foreground font-sans tracking-tight">
                                        {feat.title}
                                    </h3>
                                    <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                        {feat.desc}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ========================================================= */}
            {/* SECTION E: HOW IT WORKS */}
            {/* ========================================================= */}
            <section id="how-it-works" className="w-full py-20 sm:py-28 relative z-10">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-12">
                    <div className="text-center max-w-xl mx-auto space-y-3">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Operational Workflow
                        </span>
                        <h2 className="text-3xl sm:text-4xl md:text-5xl font-serif text-foreground">
                            Three Steps to Verification
                        </h2>
                        <p className="text-sm text-muted-foreground">
                            Streamlined verification designed for speed, reproducibility, and scientific transparency.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8 relative">
                        {[
                            {
                                step: "01",
                                title: "Upload Target Photograph",
                                desc: "Submit any JPEG, PNG, or WebP image. Client-side dimension management preserves native compression characteristics."
                            },
                            {
                                step: "02",
                                title: "Execute Forensic Engine",
                                desc: "The platform runs Error Level Analysis alongside the SigLIP vision transformer to inspect spectral and spatial feature maps."
                            },
                            {
                                step: "03",
                                title: "Inspect Findings & Certificate",
                                desc: "Review classification certainty, ELA deviation metrics, and interactive slider comparisons. Generate an official certificate."
                            }
                        ].map((step, idx) => (
                            <div
                                key={idx}
                                className="p-7 rounded-3xl bg-card border border-border shadow-sm space-y-4 relative"
                            >
                                <span className="font-mono text-3xl font-black text-primary/20">
                                    {step.step}
                                </span>
                                <h3 className="text-lg font-bold text-foreground font-sans">
                                    {step.title}
                                </h3>
                                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                    {step.desc}
                                </p>
                            </div>
                        ))}
                    </div>

                    <div className="p-4 rounded-2xl bg-muted/40 border border-border/60 text-center max-w-2xl mx-auto text-xs text-muted-foreground">
                        <p>
                            Model predictions and forensic indicators assist with technical media assessment, but do not provide an infallible guarantee of an image&apos;s real-world authenticity.
                        </p>
                    </div>
                </div>
            </section>

            {/* ========================================================= */}
            {/* SECTION F: FORENSIC ANALYSIS & TRANSPARENCY */}
            {/* ========================================================= */}
            <section id="forensics" className="w-full py-20 sm:py-28 bg-muted/30 border-t border-border/60 relative z-10">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-10">
                    <div className="text-center max-w-2xl mx-auto space-y-3">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Signal Verification Science
                        </span>
                        <h2 className="text-3xl sm:text-4xl md:text-5xl font-serif text-foreground">
                            Model Prediction vs Forensic Proof
                        </h2>
                        <p className="text-sm text-muted-foreground">
                            Understanding how AIVerifySnap separates probabilistic model output from signal-level evidence.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="p-7 rounded-3xl bg-card border border-border space-y-3 shadow-sm">
                            <div className="flex items-center gap-2">
                                <Cpu className="w-5 h-5 text-primary" />
                                <h3 className="text-base font-bold text-foreground">SigLIP Vision Transformer</h3>
                            </div>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Evaluates high-level learned feature embeddings. Produces a calibrated confidence probability reflecting how closely pixel arrangements mirror known generative models.
                            </p>
                            <span className="inline-block text-[11px] font-mono font-bold text-primary bg-primary/10 px-2.5 py-1 rounded-md">
                                Probabilistic Metric
                            </span>
                        </div>

                        <div className="p-7 rounded-3xl bg-card border border-border space-y-3 shadow-sm">
                            <div className="flex items-center gap-2">
                                <Layers className="w-5 h-5 text-primary" />
                                <h3 className="text-base font-bold text-foreground">Error Level Analysis (ELA)</h3>
                            </div>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                                Measures direct mathematical compression artifacts in the discrete cosine transform (DCT) domain. Highlights localized splicing that diverges from the global frame error baseline.
                            </p>
                            <span className="inline-block text-[11px] font-mono font-bold text-green-600 bg-green-500/10 px-2.5 py-1 rounded-md">
                                Deterministic Signal Metric
                            </span>
                        </div>
                    </div>
                </div>
            </section>

            {/* ========================================================= */}
            {/* SECTION G: PRIVACY AND TRANSPARENCY */}
            {/* ========================================================= */}
            <section id="privacy" className="w-full py-16 sm:py-20 relative z-10">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 space-y-8">
                    <div className="p-8 sm:p-10 rounded-3xl bg-card border border-border shadow-sm space-y-6">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-2xl bg-primary/10 text-primary">
                                <Lock className="w-5 h-5" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-foreground font-sans">
                                    Privacy, Processing & Data Transparency
                                </h2>
                                <p className="text-xs text-muted-foreground">
                                    Verified operational standards in AIVerifySnap
                                </p>
                            </div>
                        </div>

                        <div className="grid sm:grid-cols-2 gap-4 text-xs text-muted-foreground leading-relaxed">
                            <div className="p-4 rounded-2xl bg-muted/30 border border-border/50 space-y-1.5">
                                <span className="font-bold text-foreground">Client-Side Pre-processing</span>
                                <p>Images are scaled to maximum 512px boundaries directly in your browser prior to transmission, preserving native JPEG compression qualities.</p>
                            </div>
                            <div className="p-4 rounded-2xl bg-muted/30 border border-border/50 space-y-1.5">
                                <span className="font-bold text-foreground">Direct Stream Inference</span>
                                <p>Payloads stream directly via multipart form data to the backend detection endpoint for immediate inference execution.</p>
                            </div>
                            <div className="p-4 rounded-2xl bg-muted/30 border border-border/50 space-y-1.5">
                                <span className="font-bold text-foreground">Cryptographic SHA-256 Ledger</span>
                                <p>Scan records store the calculated hash and metadata to enable verifiable verification certificate generation.</p>
                            </div>
                            <div className="p-4 rounded-2xl bg-muted/30 border border-border/50 space-y-1.5">
                                <span className="font-bold text-foreground">User-Controlled Data</span>
                                <p>Audit histories remain accessible through the Scan History ledger with full CSV and JSON data export controls.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ========================================================= */}
            {/* SECTION H: FAQ */}
            {/* ========================================================= */}
            <section id="faq" className="w-full py-20 sm:py-24 bg-muted/20 border-t border-border/60 relative z-10">
                <div className="max-w-3xl mx-auto px-4 sm:px-6 space-y-8">
                    <div className="text-center space-y-2">
                        <span className="text-xs uppercase font-bold tracking-widest text-muted-foreground font-mono">
                            Common Questions
                        </span>
                        <h2 className="text-3xl sm:text-4xl font-serif text-foreground">
                            Frequently Asked Questions
                        </h2>
                    </div>

                    <div className="space-y-3">
                        {faqs.map((faq, idx) => (
                            <div
                                key={idx}
                                className="rounded-2xl bg-card border border-border shadow-sm overflow-hidden"
                            >
                                <button
                                    type="button"
                                    onClick={() => toggleFaq(idx)}
                                    className="w-full p-5 text-left flex items-center justify-between text-sm sm:text-base font-bold text-foreground hover:bg-muted/40 transition-colors"
                                >
                                    <span>{faq.question}</span>
                                    <ChevronDown
                                        className={`w-4 h-4 text-muted-foreground transition-transform duration-200 ${
                                            openFaq === idx ? "rotate-180 text-foreground" : ""
                                        }`}
                                    />
                                </button>
                                {openFaq === idx && (
                                    <div className="px-5 pb-5 pt-1 text-xs sm:text-sm text-muted-foreground border-t border-border/40 leading-relaxed">
                                        {faq.answer}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ========================================================= */}
            {/* SECTION I: FINAL CALL TO ACTION */}
            {/* ========================================================= */}
            <section className="w-full py-24 sm:py-32 relative z-10 overflow-hidden text-center">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 space-y-6">
                    <h2 className="text-3xl sm:text-4xl md:text-6xl font-serif text-foreground text-balance">
                        Ready to Examine Your Images?
                    </h2>
                    <p className="text-base sm:text-lg text-muted-foreground max-w-xl mx-auto leading-relaxed">
                        Start an analysis and explore the evidence available through AIVerifySnap.
                    </p>
                    <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
                        <Link
                            href="/detect"
                            className="w-full sm:w-auto px-8 py-4 bg-foreground text-background font-semibold rounded-full hover:bg-foreground/90 transition-all text-base shadow-md"
                        >
                            Start Analyzing
                        </Link>
                        <Link
                            href="/dashboard"
                            className="w-full sm:w-auto px-8 py-4 bg-card border border-border text-foreground font-semibold rounded-full hover:bg-muted transition-all text-base shadow-sm flex items-center justify-center gap-2"
                        >
                            <span>Open Dashboard</span>
                            <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        </Link>
                    </div>
                </div>

                {/* Background watermark */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full overflow-hidden flex justify-center opacity-[0.025] pointer-events-none select-none -z-10" aria-hidden="true">
                    <span className="text-[7rem] sm:text-[11rem] lg:text-[14rem] font-black tracking-tighter whitespace-nowrap">AIVERIFYSNAP</span>
                </div>
            </section>
        </div>
    );
}
