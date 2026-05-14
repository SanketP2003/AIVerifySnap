"use client";

import { motion, useInView } from "framer-motion";
import { ArrowRight, ShieldCheck, Cpu, Fingerprint, Upload, ScanLine, FileCheck, BarChart3, Users, Zap, Quote, AlertTriangle } from "lucide-react";
import Link from "next/link";
import { useRef, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { detectionApi } from "@/lib/api";

const AnimatedCounter = dynamic(
    () => Promise.resolve(({ target, suffix = "" }: { target: number; suffix?: string }) => {
        const [count, setCount] = useState(0);
        const ref = useRef<HTMLSpanElement>(null);
        const isInView = useInView(ref, { once: true });

        useEffect(() => {
            if (!isInView) return;

            let startTime: number;
            const duration = 2000;

            const animate = (time: number) => {
                if (!startTime) startTime = time;
                const progress = Math.min((time - startTime) / duration, 1);
                const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
                setCount(Math.floor(easeProgress * target));

                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    setCount(target);
                }
            };

            requestAnimationFrame(animate);
        }, [target, isInView]);

        return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
    }),
    { ssr: false }
);

const ease = [0.16, 1, 0.3, 1];

// Stats are fetched live from the backend — see useEffect in Home()

const pillars = [
    { icon: Cpu, title: "Forensic by design", desc: "Multi-layer deep feature analysis with full transparency, utilizing state-of-the-art Error Level Analysis networks." },
    { icon: ShieldCheck, title: "State-of-the-art models", desc: "Industry-leading vision models that capture the most subtle synthetic compression artifacts and generative flaws." },
    { icon: Fingerprint, title: "Protection at the core", desc: "Population-scale dataset scanning to identify unauthorized likeness use and automate digital identity takedowns." },
];

const steps = [
    { step: "01", icon: Upload, title: "Upload your media", desc: "Drag and drop an image in JPG, PNG, or WEBP. We support files up to 10MB with automatic pre-processing." },
    { step: "02", icon: ScanLine, title: "AI forensic analysis", desc: "Our dual-stream ResNet + ELA CNN analyzes compression artifacts, pixel inconsistencies, and generative signatures." },
    { step: "03", icon: FileCheck, title: "Get your verdict", desc: "Receive a detailed forensic report with confidence scores, ELA heatmaps, and explainability data." },
];

const testimonials = [
    { quote: "The ELA analysis clearly highlighted compression artifacts that were invisible to the naked eye. A promising tool for media verification workflows.", name: "Early Tester", role: "Digital Forensics Researcher", avatar: "ET" },
    { quote: "The dual-stream architecture combining RGB analysis with ELA provides a level of explainability that sets this apart from black-box classifiers.", name: "Beta User", role: "AI/ML Engineer", avatar: "BU" },
    { quote: "Being able to see the ELA heatmap alongside the original image makes it easy to explain detection results to non-technical stakeholders.", name: "Pilot User", role: "Content Moderation Lead", avatar: "PU" },
];

export default function Home() {
    const [stats, setStats] = useState([
        { value: 0, suffix: "+", label: "Images Analyzed", icon: BarChart3 },
        { value: 0, suffix: "+", label: "Registered Users", icon: Users },
        { value: 0, suffix: "", label: "Deepfakes Detected", icon: AlertTriangle },
    ]);

    useEffect(() => {
        detectionApi.getStats().then((data) => {
            setStats([
                { value: data.totalScans || 0, suffix: "+", label: "Images Analyzed", icon: BarChart3 },
                { value: data.totalUsers || 0, suffix: "+", label: "Registered Users", icon: Users },
                { value: data.deepfakesDetected || 0, suffix: "", label: "Deepfakes Detected", icon: AlertTriangle },
            ]);
        }).catch(() => { /* backend offline — keep zeros */ });
    }, []);

    return (
        <div className="flex flex-col bg-background min-h-screen relative overflow-hidden">
            {/* Ambient orbs */}
            <div className="ambient-orb ambient-orb-cool w-[700px] h-[700px] -top-[200px] -left-[200px] fixed" />
            <div className="ambient-orb ambient-orb-warm w-[500px] h-[500px] top-[300px] -right-[150px] fixed" />

            {/* ───── HERO ───── */}
            <section className="relative w-full px-6 pt-24 pb-20 lg:pt-36 lg:pb-32 flex flex-col items-center">
                <div className="text-center max-w-[820px] mx-auto relative z-10">
                    <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease }}>
                        <div className="mb-7">
                            <span className="section-pill">India&apos;s Verification Platform</span>
                        </div>
                        <h1 className="text-display-lg md:text-display-xl font-serif text-foreground text-balance mb-6">
                            Verify media. <br />
                            <span className="text-muted-foreground">Detect deepfakes.</span>
                        </h1>
                    </motion.div>

                    <motion.p
                        className="text-body-lg text-muted-foreground text-balance max-w-[580px] mx-auto mb-10"
                        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7, delay: 0.1, ease }}
                    >
                        Built on advanced forensic networks. Powered by frontier-class explainability models. Delivering irrefutable authenticity.
                    </motion.p>

                    <motion.div
                        className="flex items-center justify-center gap-3"
                        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.2, ease }}
                    >
                        <Link href="/detect" className="btn-primary text-[15px] px-7 py-3">
                            Start Verification
                        </Link>
                        <Link href="/docs" className="btn-secondary text-[15px] px-7 py-3">
                            View API Docs
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* ───── STATS ───── */}
            <section className="w-full py-14 divider-subtle" style={{ borderBottom: "1px solid hsl(var(--border) / 0.3)" }}>
                <div className="max-w-[1080px] mx-auto px-6">
                    <div className="grid grid-cols-3 gap-8">
                        {stats.map((stat, i) => (
                            <motion.div key={i} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ duration: 0.5, delay: i * 0.08, ease }}
                                className="text-center"
                            >
                                <div className="text-3xl md:text-4xl font-bold tracking-tight text-foreground font-mono tabular-nums" suppressHydrationWarning>
                                    <AnimatedCounter target={stat.value} suffix={stat.suffix} />
                                </div>
                                <div className="text-caption text-muted-foreground mt-1.5">{stat.label}</div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ───── PILLARS ───── */}
            <section className="w-full py-24 lg:py-32">
                <div className="max-w-[1120px] mx-auto px-6">
                    <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }} transition={{ duration: 0.6, ease }}
                        className="max-w-[600px] mb-16 lg:mb-20"
                    >
                        <h2 className="text-display font-serif text-foreground">
                            Powering an<br />authentic future
                        </h2>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-10 lg:gap-14">
                        {pillars.map((p, i) => (
                            <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ duration: 0.5, delay: i * 0.1, ease }}
                                className="space-y-4"
                            >
                                <div className="w-11 h-11 rounded-xl bg-foreground text-background flex items-center justify-center">
                                    <p.icon className="w-5 h-5" strokeWidth={1.8} />
                                </div>
                                <h3 className="text-heading font-semibold text-foreground">{p.title}</h3>
                                <p className="text-body text-muted-foreground leading-relaxed">{p.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ───── HOW IT WORKS ───── */}
            <section id="how-it-works" className="w-full py-24 lg:py-32 scroll-mt-20" style={{ background: "hsl(var(--muted) / 0.3)" }}>
                <div className="max-w-[960px] mx-auto px-6">
                    <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }} transition={{ duration: 0.6, ease }}
                        className="text-center mb-16 lg:mb-20"
                    >
                        <span className="section-pill mb-5 inline-flex">How it works</span>
                        <h2 className="text-display font-serif text-foreground">
                            Three steps to<br />verified truth
                        </h2>
                    </motion.div>

                    <div className="space-y-12 md:space-y-16">
                        {steps.map((step, i) => (
                            <motion.div key={i} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true, margin: "-80px" }} transition={{ duration: 0.5, delay: i * 0.08, ease }}
                                className={`flex flex-col md:flex-row items-center gap-8 md:gap-12 ${i % 2 === 1 ? "md:flex-row-reverse" : ""}`}
                            >
                                <div className="flex-1 text-center md:text-left">
                                    <span className="text-5xl font-bold text-accent/20 tracking-tighter">{step.step}</span>
                                    <h3 className="text-heading-lg font-semibold text-foreground mt-1 mb-3">{step.title}</h3>
                                    <p className="text-body text-muted-foreground leading-relaxed">{step.desc}</p>
                                </div>
                                <div className="flex-shrink-0 relative">
                                    <div className="w-20 h-20 rounded-2xl card-elevated flex items-center justify-center">
                                        <step.icon className="w-8 h-8 text-accent" strokeWidth={1.5} />
                                    </div>
                                </div>
                                <div className="flex-1 hidden md:block" />
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ───── TESTIMONIALS ───── */}
            <section className="w-full py-24 lg:py-32">
                <div className="max-w-[1120px] mx-auto px-6">
                    <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }} transition={{ duration: 0.6, ease }}
                        className="text-center mb-14"
                    >
                        <h2 className="text-display font-serif text-foreground">
                            Early access<br />feedback
                        </h2>
                        <p className="text-body text-muted-foreground mt-3">From our beta testing program</p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {testimonials.map((t, i) => (
                            <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ duration: 0.5, delay: i * 0.08, ease }}
                                className="card-elevated rounded-2xl p-7 flex flex-col"
                            >
                                <Quote className="w-6 h-6 text-accent/30 mb-4 -scale-x-100" />
                                <p className="text-[15px] text-foreground/75 leading-relaxed flex-1">
                                    &ldquo;{t.quote}&rdquo;
                                </p>
                                <div className="flex items-center gap-3 mt-6 pt-5" style={{ borderTop: "1px solid hsl(var(--border) / 0.4)" }}>
                                    <div className="w-9 h-9 rounded-full bg-accent/10 text-accent flex items-center justify-center text-[12px] font-bold">
                                        {t.avatar}
                                    </div>
                                    <div>
                                        <div className="text-[13px] font-semibold text-foreground">{t.name}</div>
                                        <div className="text-[11px] text-muted-foreground">{t.role}</div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ───── CTA ───── */}
            <section className="w-full py-28 lg:py-36 relative">
                <div className="max-w-[800px] mx-auto px-6 text-center relative z-10">
                    <motion.h2 initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }} transition={{ duration: 0.6, ease }}
                        className="text-display-lg md:text-display-xl font-serif text-foreground mb-8"
                    >
                        Build an authentic future.
                    </motion.h2>
                    <motion.div initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.1, ease }}
                        className="flex flex-wrap items-center justify-center gap-4"
                    >
                        <Link href="/detect" className="btn-primary text-[15px] px-7 py-3">
                            Try Detection Free
                        </Link>
                        <Link href="/pricing" className="inline-flex items-center gap-2 text-[15px] font-medium text-muted-foreground hover:text-foreground transition-colors group">
                            View Pricing <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </Link>
                    </motion.div>
                </div>
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full overflow-hidden flex justify-center opacity-[0.02] pointer-events-none select-none -z-10">
                    <span className="text-[14rem] font-black tracking-tighter whitespace-nowrap">AIVERIFYSNAP</span>
                </div>
            </section>
        </div>
    );
}
