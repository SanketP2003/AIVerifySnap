"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, ScanSearch, History, Shield, LayoutDashboard, Tag, Terminal } from "lucide-react";
import { useAuth, UserButton, SignInButton } from "@clerk/nextjs";
import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import { planStorage } from "@/lib/api";

const navLinks = [
    { href: "/detect", label: "Platform", icon: ScanSearch },
    { href: "/pricing", label: "Pricing", icon: Tag },
    { href: "/history", label: "History", icon: History },
    { href: "/protection", label: "Protection", icon: Shield },
    { href: "/developers", label: "Developers", icon: Terminal },
    { href: "/admin", label: "Admin", icon: LayoutDashboard },
] as const;

export function HeaderNav() {
    const { isSignedIn, isLoaded } = useAuth();
    const pathname = usePathname();
    const [mobileOpen, setMobileOpen] = useState(false);
    const [currentPlan, setCurrentPlan] = useState("PROFESSIONAL");

    useEffect(() => {
        setCurrentPlan(planStorage.getPlan());
        const handlePlanChange = () => setCurrentPlan(planStorage.getPlan());
        window.addEventListener("avs_plan_changed", handlePlanChange);
        return () => window.removeEventListener("avs_plan_changed", handlePlanChange);
    }, []);

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
        return () => { document.body.style.overflow = ""; };
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

    const isActive = (href: string) => {
        if (href === "/") return pathname === "/";
        return pathname.startsWith(href);
    };

    return (
        <>
            <header className="sticky top-0 z-40 w-full backdrop-blur-xl bg-background/80 border-b border-border/50">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6">
                    <div className="flex h-16 items-center justify-between">

                        {/* Logo */}
                        <Link
                            href="/"
                            className="flex items-center space-x-2 transition-opacity hover:opacity-70 focus-ring rounded-lg"
                            aria-label="AIVerifySnap Home"
                        >
                            <span className="font-bold tracking-tighter text-[24px] lowercase text-foreground">
                                ai<span className="font-light">verify</span>snap
                            </span>
                        </Link>

                        {/* Desktop Nav Links */}
                        <nav className="hidden lg:flex items-center gap-1" aria-label="Main navigation">
                            {navLinks.map(({ href, label }) => (
                                <Link
                                    key={href}
                                    href={href}
                                    className={cn(
                                        "px-4 py-2 text-[13px] font-semibold tracking-wider uppercase rounded-full transition-all duration-200 focus-ring",
                                        isActive(href)
                                            ? "bg-foreground text-background"
                                            : "text-foreground/70 hover:text-foreground hover:bg-muted"
                                    )}
                                    aria-current={isActive(href) ? "page" : undefined}
                                >
                                    {label}
                                </Link>
                            ))}
                        </nav>

                        {/* CTA Buttons + Auth */}
                        <div className="hidden lg:flex items-center gap-3">
                            <Link
                                href="/pricing"
                                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold hover:bg-primary/20 transition-colors"
                                title="Click to view subscription tiers"
                            >
                                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                                <span>{currentPlan}</span>
                            </Link>

                            {isLoaded && !isSignedIn && (
                                <>
                                    <SignInButton mode="modal">
                                        <button className="px-5 py-2 text-sm font-medium rounded-full bg-muted text-foreground hover:bg-muted/80 transition-colors focus-ring">
                                            Sign In
                                        </button>
                                    </SignInButton>
                                    <Link
                                        href="/sign-up"
                                        className="px-5 py-2 text-sm font-medium rounded-full bg-foreground text-background hover:bg-foreground/90 transition-all duration-200 shadow-sm focus-ring"
                                    >
                                        Get Started
                                    </Link>
                                </>
                            )}
                            {isLoaded && isSignedIn && (
                                <>
                                    <Link
                                        href="/detect"
                                        className="px-5 py-2 text-sm font-medium rounded-full bg-foreground text-background hover:bg-foreground/90 transition-all duration-200 shadow-sm focus-ring"
                                    >
                                        Experience Platform
                                    </Link>
                                    <UserButton
                                        appearance={{
                                            elements: {
                                                avatarBox: "w-9 h-9 rounded-full border-2 border-border",
                                            },
                                        }}
                                    />
                                </>
                            )}
                        </div>

                        {/* Mobile Menu Toggle */}
                        <button
                            className="lg:hidden p-2 rounded-lg text-foreground hover:bg-muted transition-colors focus-ring"
                            onClick={() => setMobileOpen(!mobileOpen)}
                            aria-expanded={mobileOpen}
                            aria-controls="mobile-nav"
                            aria-label={mobileOpen ? "Close menu" : "Open menu"}
                        >
                            {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                        </button>
                    </div>
                </div>
            </header>

            {/* Mobile Navigation Overlay */}
            {mobileOpen && (
                <div className="mobile-nav-overlay lg:hidden" id="mobile-nav" role="dialog" aria-modal="true" aria-label="Navigation menu">
                    {/* Backdrop */}
                    <div
                        className="absolute inset-0 bg-background/60 backdrop-blur-sm"
                        onClick={() => setMobileOpen(false)}
                        aria-hidden="true"
                    />

                    {/* Slide-in panel */}
                    <nav
                        className="absolute top-0 right-0 w-full max-w-sm h-full bg-background border-l border-border shadow-2xl flex flex-col animate-in slide-in-from-right duration-200"
                        aria-label="Mobile navigation"
                    >
                        {/* Mobile nav header */}
                        <div className="flex items-center justify-between p-4 border-b border-border">
                            <span className="font-bold tracking-tighter text-xl lowercase text-foreground">
                                ai<span className="font-light">verify</span>snap
                            </span>
                            <button
                                onClick={() => setMobileOpen(false)}
                                className="p-2 rounded-lg hover:bg-muted transition-colors focus-ring"
                                aria-label="Close menu"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Mobile nav links */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-1">
                            {navLinks.map(({ href, label, icon: Icon }) => (
                                <Link
                                    key={href}
                                    href={href}
                                    className={cn(
                                        "flex items-center gap-3 px-4 py-3 rounded-xl text-base font-medium transition-all duration-200 focus-ring",
                                        isActive(href)
                                            ? "bg-foreground text-background"
                                            : "text-foreground/70 hover:text-foreground hover:bg-muted"
                                    )}
                                    aria-current={isActive(href) ? "page" : undefined}
                                >
                                    <Icon className="w-5 h-5" />
                                    {label}
                                </Link>
                            ))}
                        </div>

                        {/* Mobile nav footer - auth */}
                        <div className="p-4 border-t border-border space-y-3">
                            {isLoaded && !isSignedIn && (
                                <>
                                    <SignInButton mode="modal">
                                        <button className="w-full px-5 py-3 text-sm font-medium rounded-xl bg-muted text-foreground hover:bg-muted/80 transition-colors focus-ring">
                                            Sign In
                                        </button>
                                    </SignInButton>
                                    <Link
                                        href="/sign-up"
                                        className="block w-full px-5 py-3 text-sm font-medium rounded-xl bg-foreground text-background text-center hover:bg-foreground/90 transition-all focus-ring"
                                    >
                                        Get Started
                                    </Link>
                                </>
                            )}
                            {isLoaded && isSignedIn && (
                                <div className="flex items-center justify-between">
                                    <Link
                                        href="/detect"
                                        className="flex-1 px-5 py-3 text-sm font-medium rounded-xl bg-foreground text-background text-center hover:bg-foreground/90 transition-all focus-ring"
                                    >
                                        Experience Platform
                                    </Link>
                                    <div className="ml-3">
                                        <UserButton
                                            appearance={{
                                                elements: {
                                                    avatarBox: "w-10 h-10 rounded-full border-2 border-border",
                                                },
                                            }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    </nav>
                </div>
            )}
        </>
    );
}
