"use client";

import Link from "next/link";
import { ChevronDown, Menu } from "lucide-react";
import { useAuth, UserButton, SignInButton } from "@clerk/nextjs";

export function HeaderNav() {
    const { isSignedIn, isLoaded } = useAuth();

    return (
        <header className="sticky top-0 z-40 w-full backdrop-blur-3xl bg-white/70 border-b border-black/5">
            <div className="mx-auto max-w-[1400px] px-6">
                <div className="flex h-[72px] items-center justify-between">

                    {/* Logo */}
                    <Link href="/" className="flex items-center space-x-2 transition-opacity hover:opacity-70">
                        <span className="font-bold tracking-tighter text-[26px] lowercase text-[#131313]">ai<span className="font-light">verify</span>snap</span>
                    </Link>

                    {/* Desktop Nav Links */}
                    <nav className="hidden lg:flex items-center gap-10">
                        <Link href="/detect" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                            Platform <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                        </Link>
                        <Link href="/history" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                            History <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                        </Link>
                        <Link href="/protection" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                            Protection <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                        </Link>
                        <Link href="/admin" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                            Admin <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                        </Link>
                    </nav>

                    {/* CTA Buttons + Auth */}
                    <div className="hidden lg:flex items-center gap-4">
                        {isLoaded && !isSignedIn && (
                            <>
                                <SignInButton mode="modal">
                                    <button className="flex items-center justify-center px-5 py-2.5 text-[15px] font-medium rounded-full bg-black/5 text-black hover:bg-black/10 transition-colors border border-black/5">
                                        Sign In
                                    </button>
                                </SignInButton>
                                <Link href="/sign-up" className="flex items-center justify-center px-6 py-2.5 text-[15px] font-medium rounded-full bg-[#1e1e1e] text-white hover:bg-[#2b2b2b] shadow-[inset_0_1px_4px_rgba(255,255,255,0.4)] transition-all duration-300">
                                    Get Started
                                </Link>
                            </>
                        )}
                        {isLoaded && isSignedIn && (
                            <>
                                <Link href="/detect" className="flex items-center justify-center px-6 py-2.5 text-[15px] font-medium rounded-full bg-[#1e1e1e] text-white hover:bg-[#2b2b2b] shadow-[inset_0_1px_4px_rgba(255,255,255,0.4)] transition-all duration-300">
                                    Experience Platform
                                </Link>
                                <UserButton
                                    appearance={{
                                        elements: {
                                            avatarBox: "w-10 h-10 rounded-full border-2 border-primary/20",
                                        },
                                    }}
                                />
                            </>
                        )}
                    </div>

                    {/* Mobile Menu Icon */}
                    <button className="lg:hidden p-2 text-black">
                        <Menu className="w-6 h-6" />
                    </button>
                </div>
            </div>
        </header>
    );
}
