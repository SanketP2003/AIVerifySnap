"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    ScanSearch,
    History,
    LayoutDashboard,
    Menu,
    X,
    ChevronLeft,
    ChevronRight,
    Fingerprint,
    ArrowUpRight,
    PlusCircle,
    Home,
    Cpu,
    ExternalLink,
    FileText,
    Settings
} from "lucide-react";
import { useAuth, UserButton, SignInButton } from "@clerk/nextjs";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";

interface NavItem {
    href: string;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: string;
    description?: string;
}

interface NavSection {
    title: string;
    items: NavItem[];
}

const navSections: NavSection[] = [
    {
        title: "Workspace",
        items: [
            { href: "/dashboard", label: "Overview", icon: Home, description: "Operational dashboard" },
            { href: "/history", label: "Scan History", icon: History, description: "Forensic audit ledger" },
            { href: "/reports", label: "Reports & Dossiers", icon: FileText, description: "Technical analysis reports" },
        ],
    },
    {
        title: "Forensic Tools",
        items: [
            { href: "/detect", label: "Image Authenticity", icon: ScanSearch, badge: "AI ViT", description: "SigLIP & ELA deepfake engine" },
            { href: "/protection", label: "Likeness Protection", icon: Fingerprint, badge: "Active", description: "Biometric identity monitor" },
            { href: "/admin", label: "Operations Center", icon: LayoutDashboard, badge: "Live", description: "Cluster load & threat feed" },
        ],
    },
    {
        title: "Account",
        items: [
            { href: "/settings", label: "Workspace Settings", icon: Settings, description: "Cluster config & preferences" },
        ],
    },
];

export function WorkspaceShell({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const { isSignedIn, isLoaded } = useAuth();
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);
    const [healthStatus, setHealthStatus] = useState<{ status: string; latencyMs: number } | null>(null);

    useEffect(() => {
        let mounted = true;
        import("@/lib/api").then(({ detectionApi }) => {
            detectionApi.getHealth()
                .then(h => {
                    if (mounted) setHealthStatus({ status: h.status, latencyMs: h.latencyMs ?? 0 });
                })
                .catch(() => {
                    if (mounted) setHealthStatus({ status: "UNAVAILABLE", latencyMs: 0 });
                });
        });
        return () => { mounted = false; };
    }, []);

    // Is this a public marketing route?
    const isMarketingRoute = pathname === "/" || pathname === "/features" || pathname === "/how-it-works";

    // Close mobile nav on route change
    useEffect(() => {
        setMobileOpen(false);
    }, [pathname]);

    // Lock body scroll when mobile nav is open
    useEffect(() => {
        if (mobileOpen) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }
        return () => {
            document.body.style.overflow = "";
        };
    }, [mobileOpen]);

    // Close on Escape key
    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        if (e.key === "Escape") setMobileOpen(false);
    }, []);

    useEffect(() => {
        if (mobileOpen) {
            document.addEventListener("keydown", handleKeyDown);
            return () => document.removeEventListener("keydown", handleKeyDown);
        }
    }, [mobileOpen, handleKeyDown]);

    const isRouteActive = (href: string) => {
        if (href === "/dashboard") return pathname === "/dashboard";
        if (href === "/detect") return pathname === "/detect" || pathname === "/dashboard/analyze";
        if (href === "/history") return pathname === "/history" || pathname === "/dashboard/history";
        if (href === "/reports") return pathname === "/reports" || pathname === "/dashboard/reports";
        if (href === "/settings") return pathname === "/settings" || pathname === "/dashboard/settings";
        if (href === "/") return pathname === "/";
        return pathname.startsWith(href);
    };

    // Breadcrumb title helper
    const getPageTitle = () => {
        if (pathname === "/dashboard") return "Workspace Overview";
        if (pathname === "/detect" || pathname === "/dashboard/analyze") return "Image Authenticity Verification";
        if (pathname === "/history" || pathname === "/dashboard/history") return "Forensic Scan History";
        if (pathname === "/reports" || pathname === "/dashboard/reports") return "Forensic Reports & Certificates";
        if (pathname === "/settings" || pathname === "/dashboard/settings") return "Workspace Settings";
        if (pathname === "/protection") return "Digital Likeness Protection";
        if (pathname === "/admin") return "Forensics Operations Center";
        if (pathname.startsWith("/report")) return "Verification Certificate";
        return "Workspace";
    };

    const getSectionTitle = () => {
        if (pathname === "/dashboard" || pathname === "/history" || pathname === "/reports" || pathname.startsWith("/dashboard")) return "Workspace";
        if (pathname === "/detect" || pathname === "/protection") return "Forensic Tools";
        if (pathname === "/admin") return "Forensic Tools";
        if (pathname === "/settings") return "Account";
        if (pathname.startsWith("/report")) return "Reports";
        return "AIVerifySnap";
    };

    // =========================================================
    // 1. PUBLIC MARKETING WEBSITE HEADER & NAVIGATION
    // (for /, /features, /how-it-works)
    // =========================================================
    if (isMarketingRoute) {
        return (
            <div className="min-h-screen flex flex-col bg-background text-foreground">
                {/* Top Announcement Bar */}
                <aside
                    className="w-full bg-gradient-to-r from-[#e7af55] via-[#e57a40] to-[#e7af55] py-2 flex items-center justify-center text-white text-xs sm:text-sm font-medium gap-2 px-4 shadow-sm z-50 relative"
                    aria-label="Announcement"
                >
                    <span className="bg-white/20 px-2 py-0.5 rounded-md text-[11px] font-bold border border-white/30 backdrop-blur-sm">
                        SOVEREIGN PLATFORM V2.4
                    </span>
                    <span className="hidden sm:inline">SigLIP Vision Transformer & Error Level Analysis Engine is Active</span>
                    <span className="sm:hidden">Forensic Engine Active</span>
                    <Link href="/detect" className="underline font-bold hover:opacity-90 inline-flex items-center gap-1">
                        Try Detection &rarr;
                    </Link>
                </aside>

                {/* Public Marketing Header */}
                <header className="sticky top-0 z-40 w-full backdrop-blur-xl bg-background/85 border-b border-border/60">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6">
                        <div className="flex h-16 items-center justify-between">
                            {/* Brand Logo */}
                            <Link
                                href="/"
                                className="flex items-center space-x-2 transition-opacity hover:opacity-80 focus-ring rounded-lg"
                                aria-label="AIVerifySnap Home"
                            >
                                <span className="font-bold tracking-tighter text-[24px] lowercase text-foreground">
                                    ai<span className="font-light">verify</span>snap
                                </span>
                            </Link>

                            {/* Center Marketing Navigation Links */}
                            <nav className="hidden md:flex items-center gap-1" aria-label="Main marketing navigation">
                                <Link
                                    href="/features"
                                    className={cn(
                                        "px-3.5 py-1.5 text-xs font-semibold tracking-wider uppercase rounded-full transition-colors focus-ring",
                                        pathname === "/features" ? "bg-muted text-foreground" : "text-foreground/75 hover:text-foreground hover:bg-muted"
                                    )}
                                >
                                    Features
                                </Link>
                                <Link
                                    href="/how-it-works"
                                    className={cn(
                                        "px-3.5 py-1.5 text-xs font-semibold tracking-wider uppercase rounded-full transition-colors focus-ring",
                                        pathname === "/how-it-works" ? "bg-muted text-foreground" : "text-foreground/75 hover:text-foreground hover:bg-muted"
                                    )}
                                >
                                    How It Works
                                </Link>
                                <a
                                    href={pathname === "/" ? "#forensics" : "/features"}
                                    className="px-3.5 py-1.5 text-xs font-semibold tracking-wider uppercase rounded-full text-foreground/75 hover:text-foreground hover:bg-muted transition-colors focus-ring"
                                >
                                    Forensic Engine
                                </a>
                                <a
                                    href={pathname === "/" ? "#privacy" : "/#privacy"}
                                    className="px-3.5 py-1.5 text-xs font-semibold tracking-wider uppercase rounded-full text-foreground/75 hover:text-foreground hover:bg-muted transition-colors focus-ring"
                                >
                                    Transparency
                                </a>
                                <a
                                    href={pathname === "/" ? "#faq" : "/#faq"}
                                    className="px-3.5 py-1.5 text-xs font-semibold tracking-wider uppercase rounded-full text-foreground/75 hover:text-foreground hover:bg-muted transition-colors focus-ring"
                                >
                                    FAQ
                                </a>
                            </nav>

                            {/* Right Actions: Theme, Auth, Open Workspace */}
                            <div className="flex items-center gap-2.5">
                                <ThemeToggle />

                                <div className="hidden sm:flex items-center gap-2">
                                    {isLoaded && !isSignedIn && (
                                        <>
                                            <SignInButton mode="modal">
                                                <button className="px-4 py-2 text-xs font-medium rounded-full bg-muted text-foreground hover:bg-muted/80 transition-colors focus-ring">
                                                    Sign In
                                                </button>
                                            </SignInButton>
                                            <Link
                                                href="/detect"
                                                className="px-5 py-2 text-xs font-semibold rounded-full bg-foreground text-background hover:bg-foreground/90 transition-all shadow-sm focus-ring"
                                            >
                                                Analyze an Image
                                            </Link>
                                        </>
                                    )}
                                    {isLoaded && isSignedIn && (
                                        <>
                                            <Link
                                                href="/dashboard"
                                                className="px-5 py-2 text-xs font-semibold rounded-full bg-foreground text-background hover:bg-foreground/90 transition-all shadow-sm focus-ring flex items-center gap-1.5"
                                            >
                                                <span>Open Dashboard</span>
                                                <ArrowUpRight className="w-3.5 h-3.5" />
                                            </Link>
                                            <UserButton
                                                appearance={{
                                                    elements: {
                                                        avatarBox: "w-8 h-8 rounded-full border border-border",
                                                    },
                                                }}
                                            />
                                        </>
                                    )}
                                </div>

                                {/* Mobile Hamburger */}
                                <button
                                    type="button"
                                    onClick={() => setMobileOpen(!mobileOpen)}
                                    className="md:hidden p-2 rounded-lg text-foreground hover:bg-muted transition-colors focus-ring"
                                    aria-expanded={mobileOpen}
                                    aria-label="Toggle menu"
                                >
                                    {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                                </button>
                            </div>
                        </div>
                    </div>
                </header>

                {/* Mobile Drawer for Marketing Website */}
                {mobileOpen && (
                    <div className="fixed inset-0 z-50 md:hidden" role="dialog" aria-modal="true" aria-label="Mobile Navigation">
                        <div
                            className="fixed inset-0 bg-background/80 backdrop-blur-sm"
                            onClick={() => setMobileOpen(false)}
                            aria-hidden="true"
                        />
                        <div className="fixed inset-y-0 right-0 w-full max-w-xs bg-card border-l border-border p-6 shadow-2xl flex flex-col justify-between">
                            <div className="space-y-6">
                                <div className="flex items-center justify-between pb-4 border-b border-border">
                                    <span className="font-bold text-lg tracking-tight lowercase">
                                        ai<span className="font-light">verify</span>snap
                                    </span>
                                    <button
                                        onClick={() => setMobileOpen(false)}
                                        className="p-1.5 rounded-lg hover:bg-muted"
                                        aria-label="Close menu"
                                    >
                                        <X className="w-5 h-5" />
                                    </button>
                                </div>

                                <nav className="space-y-1">
                                    <Link
                                        href="/features"
                                        onClick={() => setMobileOpen(false)}
                                        className={cn(
                                            "block px-3 py-2.5 rounded-xl text-sm font-medium transition-colors",
                                            pathname === "/features" ? "bg-muted text-foreground font-bold" : "hover:bg-muted text-foreground"
                                        )}
                                    >
                                        Features
                                    </Link>
                                    <Link
                                        href="/how-it-works"
                                        onClick={() => setMobileOpen(false)}
                                        className={cn(
                                            "block px-3 py-2.5 rounded-xl text-sm font-medium transition-colors",
                                            pathname === "/how-it-works" ? "bg-muted text-foreground font-bold" : "hover:bg-muted text-foreground"
                                        )}
                                    >
                                        How It Works
                                    </Link>
                                    <a
                                        href={pathname === "/" ? "#forensics" : "/features"}
                                        onClick={() => setMobileOpen(false)}
                                        className="block px-3 py-2.5 rounded-xl text-sm font-medium hover:bg-muted text-foreground transition-colors"
                                    >
                                        Forensic Engine
                                    </a>
                                    <a
                                        href={pathname === "/" ? "#privacy" : "/#privacy"}
                                        onClick={() => setMobileOpen(false)}
                                        className="block px-3 py-2.5 rounded-xl text-sm font-medium hover:bg-muted text-foreground transition-colors"
                                    >
                                        Privacy & Transparency
                                    </a>
                                    <a
                                        href={pathname === "/" ? "#faq" : "/#faq"}
                                        onClick={() => setMobileOpen(false)}
                                        className="block px-3 py-2.5 rounded-xl text-sm font-medium hover:bg-muted text-foreground transition-colors"
                                    >
                                        FAQ
                                    </a>
                                </nav>
                            </div>

                            <div className="pt-4 border-t border-border space-y-3">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-muted-foreground font-medium">Appearance</span>
                                    <ThemeToggle />
                                </div>
                                <Link
                                    href="/detect"
                                    onClick={() => setMobileOpen(false)}
                                    className="block w-full py-2.5 text-center bg-foreground text-background font-semibold rounded-xl text-sm shadow-sm hover:bg-foreground/90"
                                >
                                    Analyze an Image
                                </Link>
                                <Link
                                    href="/dashboard"
                                    onClick={() => setMobileOpen(false)}
                                    className="block w-full py-2.5 text-center bg-card border border-border text-foreground font-semibold rounded-xl text-sm hover:bg-muted"
                                >
                                    Open Dashboard
                                </Link>
                            </div>
                        </div>
                    </div>
                )}

                {/* Marketing Website Content */}
                <main id="main-content" className="flex-1">
                    {children}
                </main>

                {/* Shared Marketing Website Footer */}
                <footer className="w-full bg-muted/40 border-t border-border/60 py-16 relative z-10" role="contentinfo">
                    <div className="mx-auto max-w-[1400px] px-6 text-center space-y-6">
                        <h3 className="text-2xl sm:text-3xl font-serif text-foreground max-w-xl mx-auto text-balance">
                            Build the future of digital trust with AIVerifySnap.
                        </h3>
                        <p className="text-xs uppercase tracking-widest text-muted-foreground font-semibold">
                            Sovereign deepfake verification & forensic ledger
                        </p>
                        <div className="pt-8 border-t border-border/50 flex flex-col sm:flex-row items-center justify-between text-xs text-muted-foreground gap-4">
                            <Link href="/" className="font-bold tracking-tight lowercase text-foreground/75 hover:opacity-80">
                                ai<span className="font-light">verify</span>snap
                            </Link>
                            <div className="flex flex-wrap items-center justify-center gap-6">
                                <Link href="/features" className="hover:text-foreground transition-colors">Features</Link>
                                <Link href="/how-it-works" className="hover:text-foreground transition-colors">How It Works</Link>
                                <Link href="/features" className="hover:text-foreground transition-colors">Forensic Engine</Link>
                                <Link href="/#privacy" className="hover:text-foreground transition-colors">Privacy</Link>
                                <Link href="/#faq" className="hover:text-foreground transition-colors">FAQ</Link>
                                <Link href="/dashboard" className="text-foreground font-semibold hover:underline">Workspace Dashboard</Link>
                            </div>
                            <span>&copy; {new Date().getFullYear()} AIVerifySnap. All rights reserved.</span>
                        </div>
                    </div>
                </footer>
            </div>
        );
    }

    // =========================================================
    // 2. ENTERPRISE APPLICATION DASHBOARD SHELL
    // (for /dashboard, /detect, /history, /protection, /admin, /report/*)
    // =========================================================
    return (
        <div className="min-h-screen flex bg-background text-foreground antialiased selection:bg-foreground selection:text-background">
            {/* Desktop Persistent Sidebar */}
            <aside
                className={cn(
                    "hidden lg:flex flex-col border-r border-border bg-card/85 backdrop-blur-xl transition-all duration-300 z-30 sticky top-0 h-screen shrink-0",
                    sidebarCollapsed ? "w-[72px]" : "w-64"
                )}
                aria-label="Application Workspace Sidebar"
            >
                {/* Brand Header */}
                <div className="h-16 flex items-center justify-between px-4 border-b border-border shrink-0">
                    {!sidebarCollapsed ? (
                        <div className="flex items-center gap-2">
                            <Link
                                href="/"
                                className="font-bold tracking-tighter text-xl lowercase text-foreground hover:opacity-80 transition-opacity focus-ring rounded-lg"
                                title="Back to Public Website"
                            >
                                <span>ai<span className="font-light">verify</span>snap</span>
                            </Link>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground border border-border">v2.4</span>
                        </div>
                    ) : (
                        <Link href="/" className="mx-auto font-black text-lg tracking-tighter text-foreground" title="AIVerifySnap">
                            AV
                        </Link>
                    )}

                    <button
                        type="button"
                        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                        className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors focus-ring"
                        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
                        title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
                    >
                        {sidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
                    </button>
                </div>

                {/* Primary Navigation Sections */}
                <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
                    {navSections.map((section) => (
                        <div key={section.title} className="space-y-1">
                            {!sidebarCollapsed && (
                                <h3 className="px-3 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/80 mb-2">
                                    {section.title}
                                </h3>
                            )}
                            <div className="space-y-0.5">
                                {section.items.map((item) => {
                                    const active = isRouteActive(item.href);
                                    const Icon = item.icon;

                                    return (
                                        <Link
                                            key={`${section.title}-${item.label}`}
                                            href={item.href}
                                            className={cn(
                                                "group flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all duration-150 focus-ring relative",
                                                active
                                                    ? "bg-foreground text-background shadow-sm"
                                                    : "text-foreground/70 hover:text-foreground hover:bg-muted"
                                            )}
                                            title={sidebarCollapsed ? `${item.label}: ${item.description}` : undefined}
                                            aria-current={active ? "page" : undefined}
                                        >
                                            <Icon className={cn("w-4 h-4 shrink-0 transition-transform group-hover:scale-110", active ? "text-background" : "text-muted-foreground group-hover:text-foreground")} />
                                            
                                            {!sidebarCollapsed && (
                                                <div className="flex-1 flex items-center justify-between truncate">
                                                    <span className="truncate">{item.label}</span>
                                                    {item.badge && (
                                                        <span
                                                            className={cn(
                                                                "text-[10px] px-1.5 py-0.5 rounded-full font-mono",
                                                                active
                                                                    ? "bg-background/20 text-background"
                                                                    : "bg-muted text-muted-foreground border border-border/60"
                                                            )}
                                                        >
                                                            {item.badge}
                                                        </span>
                                                    )}
                                                </div>
                                            )}
                                        </Link>
                                    );
                                })}
                            </div>
                        </div>
                    ))}

                    {/* Back to Public Site Link */}
                    <div className="pt-2 border-t border-border/60">
                        <Link
                            href="/"
                            className={cn(
                                "flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors",
                                sidebarCollapsed ? "justify-center" : ""
                            )}
                            title="Return to Public Website"
                        >
                            <ExternalLink className="w-3.5 h-3.5 shrink-0" />
                            {!sidebarCollapsed && <span>Public Website</span>}
                        </Link>
                    </div>
                </div>

                {/* Inference Node Health Status Indicator */}
                {!sidebarCollapsed ? (
                    <div className="p-3 mx-3 mb-2 rounded-2xl bg-muted/40 border border-border/60 space-y-2">
                        <div className="flex items-center justify-between text-[11px]">
                            <span className="font-semibold text-foreground flex items-center gap-1.5">
                                <Cpu className="w-3.5 h-3.5 text-primary" />
                                <span>Forensics Cluster</span>
                            </span>
                            {healthStatus?.status === "UP" ? (
                                <span className="flex items-center gap-1 text-green-500 font-mono font-bold text-[10px]">
                                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                                    ONLINE
                                </span>
                            ) : healthStatus ? (
                                <span className="flex items-center gap-1 text-destructive font-mono font-bold text-[10px]">
                                    <span className="w-1.5 h-1.5 rounded-full bg-destructive" />
                                    OFFLINE
                                </span>
                            ) : (
                                <span className="flex items-center gap-1 text-muted-foreground font-mono text-[10px]">
                                    CHECKING
                                </span>
                            )}
                        </div>
                        <p className="text-[10px] text-muted-foreground leading-snug">
                            {healthStatus?.status === "UP"
                                ? `SigLIP ViT + ELA models active. Latency: ${healthStatus.latencyMs}ms`
                                : healthStatus
                                ? "Inference service unreachable. Check backend."
                                : "Connecting to analysis node..."}
                        </p>
                    </div>
                ) : (
                    <div className="p-2 mx-auto mb-2 text-center" title={healthStatus?.status === "UP" ? "Neural Cluster Online" : "Cluster Status Unavailable"}>
                        <span className={`w-2.5 h-2.5 rounded-full inline-block ${healthStatus?.status === "UP" ? "bg-green-500 animate-pulse" : "bg-muted-foreground"}`} />
                    </div>
                )}

                {/* Sidebar Footer */}
                <div className="p-3 border-t border-border flex items-center justify-between shrink-0">
                    <div className="flex items-center gap-2">
                        <ThemeToggle />
                        {!sidebarCollapsed && (
                            <span className="text-[11px] font-medium text-muted-foreground">Appearance</span>
                        )}
                    </div>

                    <div className="shrink-0">
                        {isLoaded && isSignedIn ? (
                            <UserButton
                                appearance={{
                                    elements: {
                                        avatarBox: "w-8 h-8 rounded-full border border-border",
                                    },
                                }}
                            />
                        ) : isLoaded ? (
                            <SignInButton mode="modal">
                                <button className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-muted text-foreground hover:bg-muted/80">
                                    Sign In
                                </button>
                            </SignInButton>
                        ) : null}
                    </div>
                </div>
            </aside>

            {/* Main Application Area */}
            <div className="flex-1 flex flex-col min-w-0 min-h-screen">
                {/* Top Workspace Header */}
                <header className="sticky top-0 z-20 h-16 border-b border-border/70 bg-background/85 backdrop-blur-xl px-4 sm:px-6 flex items-center justify-between gap-4">
                    {/* Left: Mobile Toggle + Breadcrumb */}
                    <div className="flex items-center gap-3 min-w-0">
                        <button
                            type="button"
                            onClick={() => setMobileOpen(true)}
                            className="lg:hidden p-2 rounded-xl text-foreground hover:bg-muted transition-colors focus-ring"
                            aria-label="Open navigation menu"
                        >
                            <Menu className="w-5 h-5" />
                        </button>

                        <div className="flex items-center gap-2 text-xs truncate">
                            <span className="text-muted-foreground hidden sm:inline">{getSectionTitle()}</span>
                            <span className="text-muted-foreground/60 hidden sm:inline">/</span>
                            <h1 className="font-bold text-foreground truncate text-sm sm:text-base font-sans">
                                {getPageTitle()}
                            </h1>
                        </div>
                    </div>

                    {/* Right: Quick actions, Telemetry Pill, Theme & Auth */}
                    <div className="flex items-center gap-2.5 shrink-0">
                        {/* Live Inference Pill */}
                        <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-full bg-card border border-border text-[11px] font-medium text-muted-foreground shadow-sm">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            <span>SigLIP Engine: Active</span>
                        </div>

                        {/* Quick New Scan Button (when not already on /detect) */}
                        {pathname !== "/detect" && (
                            <Link
                                href="/detect"
                                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-foreground text-background text-xs font-semibold hover:bg-foreground/90 transition-all shadow-sm focus-ring"
                            >
                                <PlusCircle className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">New Analysis</span>
                            </Link>
                        )}

                        <ThemeToggle className="lg:hidden" />

                        <div className="lg:hidden">
                            {isLoaded && isSignedIn && (
                                <UserButton
                                    appearance={{
                                        elements: {
                                            avatarBox: "w-8 h-8 rounded-full border border-border",
                                        },
                                    }}
                                />
                            )}
                        </div>
                    </div>
                </header>

                {/* Workspace Content Canvas */}
                <main id="main-content" className="flex-1 w-full bg-background">
                    {children}
                </main>
            </div>

            {/* Mobile Workspace Drawer */}
            {mobileOpen && (
                <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Mobile Navigation">
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 bg-background/80 backdrop-blur-sm"
                        onClick={() => setMobileOpen(false)}
                        aria-hidden="true"
                    />

                    {/* Drawer Panel */}
                    <div className="fixed inset-y-0 left-0 w-full max-w-xs bg-card border-r border-border p-5 shadow-2xl flex flex-col justify-between animate-in slide-in-from-left duration-200">
                        <div className="space-y-6">
                            {/* Drawer Header */}
                            <div className="flex items-center justify-between pb-4 border-b border-border">
                                <Link href="/dashboard" onClick={() => setMobileOpen(false)} className="font-bold text-lg tracking-tight lowercase">
                                    ai<span className="font-light">verify</span>snap
                                </Link>
                                <button
                                    onClick={() => setMobileOpen(false)}
                                    className="p-1.5 rounded-lg hover:bg-muted"
                                    aria-label="Close menu"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Drawer Nav Links */}
                            <div className="space-y-5 overflow-y-auto max-h-[70vh]">
                                {navSections.map((sec) => (
                                    <div key={sec.title} className="space-y-1">
                                        <p className="px-3 text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-1.5">
                                            {sec.title}
                                        </p>
                                        <div className="space-y-0.5">
                                            {sec.items.map((item) => {
                                                const active = isRouteActive(item.href);
                                                const Icon = item.icon;

                                                return (
                                                    <Link
                                                        key={`${sec.title}-${item.label}`}
                                                        href={item.href}
                                                        onClick={() => setMobileOpen(false)}
                                                        className={cn(
                                                            "flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors",
                                                            active
                                                                ? "bg-foreground text-background"
                                                                : "text-foreground/80 hover:bg-muted hover:text-foreground"
                                                        )}
                                                    >
                                                        <div className="flex items-center gap-3">
                                                            <Icon className={cn("w-4 h-4", active ? "text-background" : "text-muted-foreground")} />
                                                            <span>{item.label}</span>
                                                        </div>
                                                        {item.badge && (
                                                            <span
                                                                className={cn(
                                                                    "text-[10px] px-1.5 py-0.2 rounded-full font-mono",
                                                                    active ? "bg-background/20 text-background" : "bg-muted text-muted-foreground"
                                                                )}
                                                            >
                                                                {item.badge}
                                                            </span>
                                                        )}
                                                    </Link>
                                                );
                                            })}
                                        </div>
                                    </div>
                                ))}

                                <div className="pt-2 border-t border-border/60">
                                    <Link
                                        href="/"
                                        onClick={() => setMobileOpen(false)}
                                        className="flex items-center gap-3 px-3 py-2 rounded-xl text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                                    >
                                        <ExternalLink className="w-4 h-4" />
                                        <span>Back to Public Website</span>
                                    </Link>
                                </div>
                            </div>
                        </div>

                        {/* Drawer Footer */}
                        <div className="pt-4 border-t border-border space-y-3">
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-muted-foreground font-medium">Appearance</span>
                                <ThemeToggle />
                            </div>

                            {isLoaded && !isSignedIn && (
                                <SignInButton mode="modal">
                                    <button className="w-full py-2.5 text-center bg-foreground text-background text-xs font-semibold rounded-xl">
                                        Sign In
                                    </button>
                                </SignInButton>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
