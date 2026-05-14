"use client";

import { motion } from "framer-motion";
import { Check, Zap, Shield, Crown, ArrowRight, Sparkles } from "lucide-react";
import Link from "next/link";

const ease = [0.16, 1, 0.3, 1];

const tiers = [
    {
        name: "Starter",
        price: "Free",
        period: "",
        description: "For individuals exploring deepfake detection.",
        icon: Zap,
        iconBg: "bg-blue-500/8 text-blue-600 dark:text-blue-400",
        cta: "Get Started Free",
        ctaClass: "btn-secondary",
        features: [
            "10 image scans per day",
            "Basic ELA analysis",
            "Scan history (7 days)",
            "Standard processing speed",
            "Community support",
        ],
    },
    {
        name: "Professional",
        price: "₹1,499",
        period: "/month",
        description: "For creators, journalists, and researchers.",
        icon: Shield,
        iconBg: "bg-accent/10 text-accent",
        cta: "Start Free Trial",
        ctaClass: "btn-primary",
        popular: true,
        features: [
            "Unlimited image scans",
            "Advanced ELA + heatmap analysis",
            "Full scan history",
            "Bulk upload (up to 50 images)",
            "PDF report export",
            "Identity protection scans",
            "Priority processing",
            "Email support",
        ],
    },
    {
        name: "Enterprise",
        price: "Custom",
        period: "",
        description: "For organizations and platforms at scale.",
        icon: Crown,
        iconBg: "bg-orange-500/8 text-orange-600 dark:text-orange-400",
        cta: "Contact Sales",
        ctaClass: "btn-secondary",
        features: [
            "Everything in Professional",
            "REST API access with keys",
            "Video frame analysis",
            "Custom model fine-tuning",
            "Bulk upload (unlimited)",
            "SLA guarantee (99.9% uptime)",
            "Webhook notifications",
            "Dedicated account manager",
            "On-premise deployment option",
        ],
    },
];

const faqs = [
    { q: "Can I switch plans later?", a: "Yes, you can upgrade or downgrade at any time. Changes take effect immediately and billing is prorated." },
    { q: "Is there a free trial for Professional?", a: "Absolutely. Every Professional plan starts with a 14-day free trial, no credit card required." },
    { q: "What counts as a scan?", a: "Each image analyzed counts as one scan. Bulk uploads count each file individually. Viewing history or reports does not consume scans." },
    { q: "Do you offer student or non-profit discounts?", a: "Yes! We offer 50% off Professional plans for verified students and registered non-profits. Contact us for details." },
];

export default function PricingPage() {
    return (
        <div className="flex flex-col items-center bg-background min-h-screen relative overflow-hidden">
            <div className="ambient-orb ambient-orb-cool w-[500px] h-[500px] -top-[100px] -left-[150px] absolute" />
            <div className="ambient-orb ambient-orb-warm w-[400px] h-[400px] top-[300px] -right-[100px] absolute" />

            {/* Hero */}
            <section className="w-full px-6 pt-20 pb-14 text-center relative z-10">
                <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.7, ease }} className="max-w-[640px] mx-auto space-y-5">
                    <span className="section-pill">
                        <Sparkles className="w-3.5 h-3.5 text-accent" />
                        Simple, transparent pricing
                    </span>
                    <h1 className="text-display-lg font-serif text-foreground">
                        Plans for every<br /><span className="text-muted-foreground">verification need.</span>
                    </h1>
                    <p className="text-body-lg text-muted-foreground max-w-[480px] mx-auto">
                        Start free. Upgrade when you need advanced forensics, bulk uploads, or API access.
                    </p>
                </motion.div>
            </section>

            {/* Cards */}
            <section className="w-full max-w-[1080px] mx-auto px-6 pb-24 relative z-10">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                    {tiers.map((tier, i) => (
                        <motion.div key={tier.name}
                            initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: i * 0.08, ease }}
                            className={`relative flex flex-col rounded-2xl p-7 card-elevated ${
                                tier.popular ? "ring-1 ring-accent/30" : ""
                            }`}
                        >
                            {tier.popular && (
                                <div className="absolute -top-3.5 left-1/2 -translate-x-1/2">
                                    <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider text-white"
                                        style={{ background: "hsl(var(--accent))", boxShadow: "0 2px 12px hsl(var(--accent) / 0.3)" }}>
                                        <Sparkles className="w-3 h-3" /> Most Popular
                                    </span>
                                </div>
                            )}

                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${tier.iconBg} mb-5`}>
                                <tier.icon className="w-5 h-5" strokeWidth={1.8} />
                            </div>

                            <h3 className="text-lg font-bold tracking-tight text-foreground">{tier.name}</h3>
                            <p className="text-caption text-muted-foreground mt-1 mb-5">{tier.description}</p>

                            <div className="flex items-baseline gap-1 mb-7">
                                <span className="text-4xl font-bold tracking-tight text-foreground">{tier.price}</span>
                                {tier.period && <span className="text-caption text-muted-foreground">{tier.period}</span>}
                            </div>

                            <Link href={tier.name === "Enterprise" ? "/#contact" : "/sign-up"}
                                className={`${tier.ctaClass} w-full py-3 text-[14px] mb-7 justify-center`}>
                                {tier.cta} <ArrowRight className="w-3.5 h-3.5" />
                            </Link>

                            <div className="space-y-3 flex-1">
                                {tier.features.map((f) => (
                                    <div key={f} className="flex items-start gap-2.5">
                                        <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" strokeWidth={2} />
                                        <span className="text-[13px] text-muted-foreground leading-snug">{f}</span>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* FAQ */}
            <section className="w-full max-w-[640px] mx-auto px-6 pb-32 relative z-10">
                <motion.h2 initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }} transition={{ duration: 0.5, ease }}
                    className="text-heading-lg font-serif text-foreground text-center mb-10">
                    Frequently asked questions
                </motion.h2>
                <div className="space-y-3">
                    {faqs.map((faq, i) => (
                        <motion.div key={i} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.05, ease }}
                            className="card-elevated rounded-xl p-5">
                            <h3 className="text-[14px] font-semibold text-foreground">{faq.q}</h3>
                            <p className="text-caption text-muted-foreground mt-1.5 leading-relaxed">{faq.a}</p>
                        </motion.div>
                    ))}
                </div>
            </section>
        </div>
    );
}
