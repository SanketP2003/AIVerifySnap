package com.backend.aiverifysnap.dto;


public class UserCreationDto {

    private Long id;
    private String name;
    private String email;
    private String role;
    private String clerkId;

    public UserCreationDto() {}

    public UserCreationDto(Long id, String name, String email, String role, String clerkId) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.role = role;
        this.clerkId = clerkId;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public String getClerkId() {
        return clerkId;
    }

    public void setClerkId(String clerkId) {
        this.clerkId = clerkId;
    }

}