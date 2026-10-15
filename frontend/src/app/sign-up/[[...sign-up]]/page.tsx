"use client";

import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
    return (
        <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center p-4">
            <SignUp
                appearance={{
                    elements: {
                        rootBox: "mx-auto",
                        card: "rounded-2xl shadow-lg border border-border/80",
                    },
                }}
            />
        </div>
    );
}
