"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
    Check,
    X,
    Shield,
    Building2,
    HelpCircle,
    BadgeCheck
} from "lucide-react";
import { planStorage, detectionApi } from "@/lib/api";
import { toast } from "sonner";
import { useUser } from "@clerk/nextjs";

export default function PricingPage() {
    const { user } = useUser();
    const [billingCycle, setBillingCycle] = useState<"monthly" | "annual">("monthly");
    const [currentPlan, setCurrentPlan] = useState<"STARTER" | "PROFESSIONAL" | "ENTERPRISE">("PROFESSIONAL");
    const [enterpriseModalOpen, setEnterpriseModalOpen] = useState(false);
    const [contactEmail, setContactEmail] = useState("");
    const [contactOrg, setContactOrg] = useState("");
    const [submittedContact, setSubmittedContact] = useState(false);

    useEffect(() => {
        setCurrentPlan(planStorage.getPlan());
        const handlePlanChange = () => setCurrentPlan(planStorage.getPlan());
        window.addEventListener("avs_plan_changed", handlePlanChange);
        return () => window.removeEventListener("avs_plan_changed", handlePlanChange);
    }, []);

    const handleSelectPlan = async (plan: "STARTER" | "PROFESSIONAL" | "ENTERPRISE") => {
        if (plan === "ENTERPRISE") {
            setEnterpriseModalOpen(true);
            return;
        }

        const username = user?.username || user?.id || "guest";
        planStorage.setPlan(plan);
        setCurrentPlan(plan);

        try {
            await detectionApi.updateUserPlan(username, plan);
        } catch {
            // locally updated
        }

        toast.success(`Plan updated to ${plan === "PROFESSIONAL" ? "Professional (₹1,499/mo)" : "Starter (Free)"}!`, {
            description: plan === "PROFESSIONAL" 
                ? "Unlocked: Bulk upload up to 50 images, Identity Likeness Protection & PDF export." 
                : "Switched to Starter Plan (10 scans/day limit)."
        });
    };

    const handleEnterpriseSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setSubmittedContact(true);
        planStorage.setPlan("ENTERPRISE");
        setCurrentPlan("ENTERPRISE");
        toast.success("Enterprise trial access unlocked!", {
            description: "Custom model parameters, unlimited bulk uploads and REST API keys are now active."
        });
        setTimeout(() => {
            setEnterpriseModalOpen(false);
            setSubmittedContact(false);
        }, 1500);
    };

    const plans = [
        {
            id: "STARTER" as const,
            name: "Starter",
            tagline: "Essential verification for casual creators & individual researchers.",
            priceMonthly: "₹0",
            priceAnnual: "₹0",
            period: "forever free",
            highlight: false,
            badge: "Free Tier",
            features: [
                "10 image scans per day",
                "Basic Error Level Analysis (ELA)",
                "7-day scan history retention",
                "Single image upload queue",
                "Community support",
                "Standard web app access",
            ],
            missingFeatures: [
                "Bulk multi-image scanning (up to 50)",
                "Cryptographic PDF forensic reports",
                "Biometric identity likeness protection",
                "Full unlimited scan history",
                "REST API keys & Webhook access",
                "Video frame-by-frame forensics",
            ],
            buttonText: currentPlan === "STARTER" ? "Current Plan" : "Downgrade to Starter",
            buttonVariant: "outline"
        },
        {
            id: "PROFESSIONAL" as const,
            name: "Professional",
            tagline: "Advanced forensic suite for media teams, fact-checkers, & journalists.",
            priceMonthly: "₹1,499",
            priceAnnual: "₹1,199",
            period: billingCycle === "monthly" ? "per month" : "per month, billed annually",
            highlight: true,
            badge: "Most Popular",
            features: [
                "Unlimited daily image scans",
                "Advanced ELA & spatial heatmaps",
                "Bulk scan up to 50 images simultaneously",
                "Full permanent scan history",
                "Biometric digital likeness protection",
                "Cryptographic PDF forensic report export",
                "Google Lens & reverse visual search",
                "Priority GPU neural inference queue",
            ],
            missingFeatures: [
                "REST API developer keys",
                "Video frame-by-frame forensics",
                "Custom fine-tuned weights",
            ],
            buttonText: currentPlan === "PROFESSIONAL" ? "Active Plan" : "Upgrade to Professional",
            buttonVariant: "solid"
        },
        {
            id: "ENTERPRISE" as const,
            name: "Enterprise",
            tagline: "Mission-critical defense for newsrooms, legal firms, & law enforcement.",
            priceMonthly: "Custom",
            priceAnnual: "Custom",
            period: "tailored SLA & volume",
            highlight: false,
            badge: "Enterprise Grade",
            features: [
                "Everything in Professional",
                "Unlimited bulk upload queues",
                "Video frame-by-frame deepfake analysis",
                "REST API keys & automated webhooks",
                "Custom model fine-tuning & domain adaptation",
                "Dedicated enterprise SLA & 99.9% uptime",
                "On-premises / air-gapped Docker deployment",
                "Dedicated forensic compliance engineer",
            ],
            missingFeatures: [],
            buttonText: currentPlan === "ENTERPRISE" ? "Active Plan" : "Request Enterprise Access",
            buttonVariant: "enterprise"
        }
    ];

    const comparisonFeatures = [
        { category: "Scanning & Detection", items: [
            { name: "Daily scan limit", starter: "10 scans / day", pro: "Unlimited", ent: "Unlimited" },
            { name: "Bulk multi-image scanning", starter: "1 image at a time", pro: "Up to 50 images", ent: "Unlimited batch" },
            { name: "Error Level Analysis (ELA)", starter: "Standard delta", pro: "High-res heatmap fusion", ent: "Custom quantization" },
            { name: "Dual-Stream Neural Model", starter: "SigLIP fallback", pro: "Dual-Stream ResNet18 + ELA", ent: "Custom fine-tuned" },
            { name: "Video frame analysis", starter: "Not available", pro: "Add-on queue", ent: "Full 60fps frame audit" },
        ]},
        { category: "Forensics & Identity", items: [
            { name: "Forensic PDF dossier export", starter: "JSON only", pro: "Certified Cryptographic PDF", ent: "PDF + Courtroom Evidence Package" },
            { name: "Scan history retention", starter: "7 days", pro: "Unlimited permanent", ent: "Unlimited with audit logs" },
            { name: "Biometric Likeness Protection", starter: "Preview only", pro: "Full web & forum scan + DMCA notice", ent: "Real-time automated web crawler" },
            { name: "Google Lens reverse matching", starter: "Basic", pro: "Neural visual fingerprinting", ent: "Automated reverse provenance" },
        ]},
        { category: "Developer & Integration", items: [
            { name: "REST API Access", starter: "Not included", pro: "Interactive Docs only", ent: "Full REST API keys (high TPS)" },
            { name: "Webhooks for scan alerts", starter: "Not included", pro: "Not included", ent: "Real-time webhook pings" },
            { name: "Deployment model", starter: "Cloud SaaS", pro: "Cloud SaaS", ent: "Cloud / On-premises / VPC" },
            { name: "Support tier", starter: "Community", pro: "Priority email & ticket", ent: "Dedicated 24/7 Slack / SLA" },
        ]}
    ];

    return (
        <div className="min-h-screen bg-background text-foreground pb-24">
            {/* Background ambient gradient */}
            <div className="relative overflow-hidden pt-12 pb-16 md:pt-20 md:pb-24">
                <div className="absolute inset-0 -z-10 flex items-center justify-center opacity-30 pointer-events-none">
                    <div className="w-[600px] h-[600px] bg-primary/20 blur-[130px] rounded-full" />
                </div>

                <div className="max-w-[1200px] mx-auto px-4 sm:px-6 text-center space-y-6">
                    <motion.div
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-xs font-semibold uppercase tracking-wider text-primary"
                    >
                        <Shield className="w-3.5 h-3.5" />
                        Transparent, Value-Driven Subscriptions
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-4xl sm:text-5xl md:text-6xl font-black tracking-tight"
                    >
                        Forensic-Grade Detection. <br className="hidden sm:inline" />
                        <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                            Flexible For Every Scale.
                        </span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="text-muted-foreground text-base sm:text-lg max-w-2xl mx-auto"
                    >
                        Defend your organisation and identity against malicious synthetic media. From single scans to 50-image batch audits and enterprise REST API pipelines.
                    </motion.p>

                    {/* Billing Cycle Switcher */}
                    <div className="flex items-center justify-center gap-3 pt-4">
                        <div className="inline-flex items-center p-1 rounded-full bg-muted border border-border">
                            <button
                                onClick={() => setBillingCycle("monthly")}
                                className={`px-5 py-2 text-xs sm:text-sm font-semibold rounded-full transition-all ${
                                    billingCycle === "monthly"
                                        ? "bg-foreground text-background shadow-sm"
                                        : "text-muted-foreground hover:text-foreground"
                                }`}
                            >
                                Monthly Billing
                            </button>
                            <button
                                onClick={() => setBillingCycle("annual")}
                                className={`px-5 py-2 text-xs sm:text-sm font-semibold rounded-full transition-all flex items-center gap-1.5 ${
                                    billingCycle === "annual"
                                        ? "bg-foreground text-background shadow-sm"
                                        : "text-muted-foreground hover:text-foreground"
                                }`}
                            >
                                Annual Billing
                                <span className="px-1.5 py-0.5 text-[10px] font-bold rounded-full bg-emerald-500/20 text-emerald-400">
                                    Save 20%
                                </span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Cards Grid */}
            <div className="max-w-[1240px] mx-auto px-4 sm:px-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-stretch">
                    {plans.map((p) => {
                        const isCurrent = currentPlan === p.id;
                        return (
                            <motion.div
                                key={p.id}
                                whileHover={{ y: -4 }}
                                transition={{ duration: 0.2 }}
                                className={`relative flex flex-col justify-between rounded-3xl p-6 sm:p-8 transition-all ${
                                    p.highlight
                                        ? "bg-card border-2 border-primary shadow-xl shadow-primary/10 ring-1 ring-primary/20"
                                        : "bg-card/70 border border-border shadow-sm hover:border-border/80"
                                }`}
                            >
                                {p.highlight && (
                                    <div className="absolute -top-3.5 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full bg-primary text-primary-foreground text-xs font-bold uppercase tracking-wider shadow-md">
                                        {p.badge}
                                    </div>
                                )}

                                <div className="space-y-6">
                                    {/* Plan Header */}
                                    <div className="space-y-2">
                                        <div className="flex items-center justify-between">
                                            <h3 className="text-xl font-bold">{p.name}</h3>
                                            {isCurrent && (
                                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
                                                    <BadgeCheck className="w-3.5 h-3.5" />
                                                    Active
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-xs text-muted-foreground min-h-[36px]">{p.tagline}</p>
                                    </div>

                                    {/* Price Box */}
                                    <div className="pt-2 pb-4 border-b border-border/60">
                                        <div className="flex items-baseline gap-2">
                                            <span className="text-4xl sm:text-5xl font-black tracking-tight">
                                                {billingCycle === "monthly" ? p.priceMonthly : p.priceAnnual}
                                            </span>
                                            <span className="text-xs text-muted-foreground font-medium">
                                                {p.period}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Features List */}
                                    <div className="space-y-3 pt-2">
                                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                                            Included Capabilities
                                        </span>
                                        <ul className="space-y-2.5 text-xs sm:text-sm">
                                            {p.features.map((feat, idx) => (
                                                <li key={idx} className="flex items-start gap-2.5">
                                                    <div className="p-0.5 rounded-full bg-emerald-500/10 text-emerald-500 mt-0.5 shrink-0">
                                                        <Check className="w-3.5 h-3.5" />
                                                    </div>
                                                    <span>{feat}</span>
                                                </li>
                                            ))}
                                            {p.missingFeatures.map((mFeat, idx) => (
                                                <li key={idx} className="flex items-start gap-2.5 text-muted-foreground/50">
                                                    <div className="p-0.5 rounded-full bg-muted text-muted-foreground/40 mt-0.5 shrink-0">
                                                        <X className="w-3.5 h-3.5" />
                                                    </div>
                                                    <span className="line-through">{mFeat}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>

                                {/* Action Button */}
                                <div className="pt-8">
                                    <button
                                        onClick={() => handleSelectPlan(p.id)}
                                        className={`w-full py-3 px-6 rounded-2xl font-bold text-sm transition-all focus-ring ${
                                            p.highlight
                                                ? "bg-foreground text-background hover:bg-foreground/90 shadow-md"
                                                : isCurrent
                                                ? "bg-muted text-muted-foreground border border-border"
                                                : "bg-muted hover:bg-muted/80 text-foreground border border-border/80"
                                        }`}
                                    >
                                        {isCurrent ? "Current Plan Active" : p.buttonText}
                                    </button>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Feature Comparison Matrix */}
            <div className="max-w-[1200px] mx-auto px-4 sm:px-6 pt-24 space-y-8">
                <div className="text-center space-y-2">
                    <h2 className="text-2xl sm:text-3xl font-black">Detailed Plan Capabilities</h2>
                    <p className="text-muted-foreground text-sm max-w-xl mx-auto">
                        Examine forensic capabilities, algorithmic pipelines, and enterprise compliance across all tiers.
                    </p>
                </div>

                <div className="bg-card border border-border rounded-3xl overflow-hidden shadow-sm">
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs sm:text-sm">
                            <thead>
                                <tr className="border-b border-border bg-muted/40">
                                    <th className="py-4 px-6 font-bold text-muted-foreground uppercase tracking-wider w-2/5">
                                        Feature Examination
                                    </th>
                                    <th className="py-4 px-4 font-bold text-foreground w-1/5 text-center">Starter</th>
                                    <th className="py-4 px-4 font-bold text-primary w-1/5 text-center">Professional</th>
                                    <th className="py-4 px-4 font-bold text-foreground w-1/5 text-center">Enterprise</th>
                                </tr>
                            </thead>
                            <tbody>
                                {comparisonFeatures.map((sec, secIdx) => (
                                    <div key={secIdx} className="contents">
                                        <tr className="bg-muted/20 border-b border-border/60">
                                            <td colSpan={4} className="py-2.5 px-6 font-bold text-primary text-xs uppercase tracking-wider">
                                                {sec.category}
                                            </td>
                                        </tr>
                                        {sec.items.map((it, itIdx) => (
                                            <tr key={itIdx} className="border-b border-border/40 hover:bg-muted/10 transition-colors">
                                                <td className="py-3.5 px-6 font-medium text-foreground">{it.name}</td>
                                                <td className="py-3.5 px-4 text-center text-muted-foreground">{it.starter}</td>
                                                <td className="py-3.5 px-4 text-center font-semibold text-foreground">{it.pro}</td>
                                                <td className="py-3.5 px-4 text-center text-muted-foreground">{it.ent}</td>
                                            </tr>
                                        ))}
                                    </div>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* FAQ Section */}
            <div className="max-w-3xl mx-auto px-4 sm:px-6 pt-20 space-y-8">
                <div className="text-center space-y-2">
                    <h2 className="text-2xl sm:text-3xl font-black">Frequently Asked Questions</h2>
                    <p className="text-muted-foreground text-sm">
                        Everything you need to know about scanning quotas, verification methods, and forensic reports.
                    </p>
                </div>

                <div className="space-y-4">
                    {[
                        {
                            q: "How does the 50-image bulk scan work on the Professional plan?",
                            a: "On the Professional plan, simply drop up to 50 images into the scanner simultaneously. Our parallel worker queue processes each image through Error Level Analysis and SigLIP Vision Transformer inference, displaying live batch stats and enabling 1-click batch PDF export."
                        },
                        {
                            q: "What is included in the certified PDF Forensic Report?",
                            a: "Every export includes a verifiable SHA-256 cryptographic hash, ELA quantization metrics (mean, std dev, max delta), classification verdict, confidence breakdown, examiner stamp, and chain of custody documentation."
                        },
                        {
                            q: "How does Digital Likeness Identity Protection work?",
                            a: "Identity Protection allows you to upload your reference portrait. We extract non-reversible facial feature maps and scan public imageboards, deepfake forums, and generative datasets for unauthorized clones, and generate legal DMCA takedown notice letters."
                        },
                        {
                            q: "How can developers integrate with the REST API?",
                            a: "Enterprise subscribers receive dedicated API keys with endpoints for synchronous single image detection, bulk batch scanning, reverse image searching, and historical retrieval. Interactive docs are available at /developers."
                        }
                    ].map((item, idx) => (
                        <div key={idx} className="p-5 rounded-2xl bg-card border border-border space-y-2">
                            <h4 className="font-bold text-sm sm:text-base flex items-center gap-2">
                                <HelpCircle className="w-4 h-4 text-primary shrink-0" />
                                {item.q}
                            </h4>
                            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed pl-6">
                                {item.a}
                            </p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Enterprise Consultation Modal */}
            {enterpriseModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-md">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-card border border-border p-6 sm:p-8 rounded-3xl max-w-md w-full shadow-2xl space-y-6"
                    >
                        <div className="space-y-1">
                            <h3 className="text-xl font-bold flex items-center gap-2">
                                <Building2 className="w-5 h-5 text-primary" />
                                Enterprise Solution Inquiry
                            </h3>
                            <p className="text-xs text-muted-foreground">
                                Access REST API keys, video frame inference, custom model weights, and on-premises Docker containers.
                            </p>
                        </div>

                        {!submittedContact ? (
                            <form onSubmit={handleEnterpriseSubmit} className="space-y-4">
                                <div>
                                    <label className="text-xs font-semibold text-muted-foreground block mb-1">Corporate Email</label>
                                    <input
                                        type="email"
                                        required
                                        value={contactEmail}
                                        onChange={(e) => setContactEmail(e.target.value)}
                                        placeholder="name@organization.com"
                                        className="w-full px-4 py-2.5 rounded-xl bg-muted border border-border text-sm focus-ring"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs font-semibold text-muted-foreground block mb-1">Organization / Newsroom</label>
                                    <input
                                        type="text"
                                        required
                                        value={contactOrg}
                                        onChange={(e) => setContactOrg(e.target.value)}
                                        placeholder="Media Corp / Legal Department"
                                        className="w-full px-4 py-2.5 rounded-xl bg-muted border border-border text-sm focus-ring"
                                    />
                                </div>
                                <div className="flex gap-3 pt-2">
                                    <button
                                        type="button"
                                        onClick={() => setEnterpriseModalOpen(false)}
                                        className="flex-1 py-2.5 rounded-xl border border-border text-xs font-semibold hover:bg-muted"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="submit"
                                        className="flex-1 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold hover:bg-foreground/90"
                                    >
                                        Unlock Enterprise
                                    </button>
                                </div>
                            </form>
                        ) : (
                            <div className="py-6 text-center space-y-2">
                                <div className="inline-flex p-3 rounded-full bg-emerald-500/10 text-emerald-500">
                                    <Check className="w-6 h-6" />
                                </div>
                                <h4 className="font-bold">Enterprise Tier Enabled</h4>
                                <p className="text-xs text-muted-foreground">
                                    API keys, unlimited batch upload, and video analysis are now active on your session.
                                </p>
                            </div>
                        )}
                    </motion.div>
                </div>
            )}
        </div>
    );
}
