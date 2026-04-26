package com.backend.aiverifysnap.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserCreationDto {

    private Long id;
    private String name;
    private String email;
    private String role;
    private String clerkId;
}
